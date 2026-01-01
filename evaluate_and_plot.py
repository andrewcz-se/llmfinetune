import matplotlib.pyplot as plt
import json
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import datetime

# --- Configuration ---
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "Qwen2.5-3B-Clinical-Extract"
REPORT_FILENAME = "evaluation_results.md"

# --- Part 1: Plotting Loss Curves ---
def plot_loss_curve(log_history):
    train_loss = []
    steps = []
    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            train_loss.append(entry['loss'])
            steps.append(entry['step'])

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss, label='Training Loss', color='blue')
    plt.title('Training Dynamics')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Loss curve saved as 'loss_curve.png'")

try:
    checkpoints = sorted(glob.glob("./results/checkpoint-*"))
    if checkpoints:
        with open(os.path.join(checkpoints[-1], "trainer_state.json"), "r") as f:
            data = json.load(f)
            plot_loss_curve(data['log_history'])
except Exception as e:
    print(f"Skipping plot: {e}")

# --- Part 2: The Test Suite ---

TEST_SUITE = [
    {
        "name": "Standard Case",
        "note": "Pt John Doe, 55y, history of T2DM. Meds: Metformin 500mg."
    },
    {
        "name": "Missing Age (Null Check)",
        "note": "Patient Sarah Connor presents with severe anxiety. Rx: Diazepam 5mg.",
        "check_null_age": True
    },
    {
        "name": "Multi-Medication 1 (Array Check)",
        "note": "Pt Paul Bunyon, 48 yrs, reports fever. Prescribe: Ibuprofen 400mg and Paracetamol 500mg.",
        "check_multi_med": True
    },
    {
        "name": "Multi-Medication 2 (Array Check)",
        "note": "Pt Dave Smith, 28y, reports headache. Rx: Codeine 400mg and Paracetamol 500mg.",
        "check_multi_med": True
    },
    {
        "name": "Enhanced Standard Case",
        "note": "Pete Jones (68), presents diarrhea. Dx gastroenteritis. Prescribe Loperamide 20mg, Gaviscon 250ml, Omprazole 50mg",
        
    }
]

def generate_response(model, tokenizer, text):
    messages = [
        {"role": "system", "content": "Extract clinical data into strict Solaris-compliant JSON."},
        {"role": "user", "content": text}
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=300, temperature=0.1)
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def clean_for_md(text):
    """Removes existing markdown fences to prevent rendering errors in the report."""
    return text.replace("```json", "").replace("```", "").strip()

def analyze_output(json_str, test_case):
    try:
        data = json.loads(json_str)
        if data.get("resourceType") != "SolarisEMRClinicalSummary":
            return {"status": "FAIL", "detail": "Wrong ResourceType"}
        
        subj = data.get("subject", {})
        age_val = subj.get("ageInt", "MISSING")
        if test_case.get("check_null_age") and age_val is not None:
             return {"status": "FAIL", "detail": f"Hallucinated Age: {age_val}"}

        meds = data.get("medicationRequest", [])
        if test_case.get("check_multi_med") and len(meds) != 2:
             return {"status": "FAIL", "detail": f"Expected 2 meds, found {len(meds)}"}

        if len(meds) > 0:
            try:
                dose = meds[0]["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]
                if "value" not in dose or "unit" not in dose:
                    return {"status": "PARTIAL", "detail": "Missing unit split"}
            except:
                return {"status": "PARTIAL", "detail": "Deep nesting failed"}
        
        return {"status": "PASS", "detail": "Perfect"}
            
    except json.JSONDecodeError:
        return {"status": "FAIL", "detail": "Invalid JSON"}

# --- Execution & Logging ---

results_log = []

print("\n=== 1. Loading BASE Model ===")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    quantization_config=bnb_config, 
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print("\n--- Running Base Model Tests ---")
for case in TEST_SUITE:
    print(f"Processing: {case['name']}...")
    output = generate_response(base_model, tokenizer, case['note'])
    analysis = analyze_output(output, case)
    results_log.append({
        "case": case,
        "model": "Base Model (Qwen 2.5 3B)",
        "output": output,
        "analysis": analysis
    })

print("\n=== 2. Loading ADAPTER (Fine-Tuned) ===")
model_with_adapter = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("\n--- Running Fine-Tuned Model Tests ---")
for case in TEST_SUITE:
    print(f"Processing: {case['name']}...")
    output = generate_response(model_with_adapter, tokenizer, case['note'])
    analysis = analyze_output(output, case)
    results_log.append({
        "case": case,
        "model": "Fine-Tuned (Clinical-Extract)",
        "output": output,
        "analysis": analysis
    })

# --- Write Report to File ---
print(f"\nWriting detailed report to {REPORT_FILENAME}...")

with open(REPORT_FILENAME, "w", encoding="utf-8") as f:
    f.write(f"# Clinical Extraction Model Evaluation Report\n")
    f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 1. Summary Table
    f.write("## 1. Summary of Results\n")
    f.write("| Test Case | Model | Status | Detail |\n")
    f.write("| :--- | :--- | :--- | :--- |\n")
    for res in results_log:
        status_icon = "✅" if res["analysis"]["status"] == "PASS" else "❌"
        if res["analysis"]["status"] == "PARTIAL": status_icon = "⚠️"
        f.write(f"| {res['case']['name']} | {res['model']} | {status_icon} {res['analysis']['status']} | {res['analysis']['detail']} |\n")
    
    # 2. Detailed Breakdown
    f.write("\n## 2. Detailed Input/Output Logs\n")
    
    # Group by Test Case for side-by-side comparison logic
    for i in range(len(TEST_SUITE)):
        base_res = results_log[i] # First half of log is base
        tuned_res = results_log[i + len(TEST_SUITE)] # Second half is tuned
        
        f.write(f"\n### Test Case: {base_res['case']['name']}\n")
        f.write(f"**Input Note:**\n> {base_res['case']['note']}\n\n")
        
        f.write("#### Base Model Output\n")
        f.write(f"```json\n{clean_for_md(base_res['output'])}\n```\n")
        f.write(f"*Analysis: {base_res['analysis']['status']} - {base_res['analysis']['detail']}*\n\n")
        
        f.write("#### Fine-Tuned Model Output\n")
        f.write(f"```json\n{clean_for_md(tuned_res['output'])}\n```\n")
        f.write(f"*Analysis: {tuned_res['analysis']['status']} - {tuned_res['analysis']['detail']}*\n")
        f.write("\n---\n")

print("Evaluation completed, check generated report and loss curve.")