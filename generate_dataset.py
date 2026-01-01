import json
import random
import os

# Configuration
NUM_SAMPLES = 500

# --- POOL 1: TRAINING DATA ---
TRAIN_CONDITIONS = [
    "Type 2 diabetes", "Hypertension", "Migraine", 
    "Gastroesophageal reflux", "Anxiety disorder"
]
TRAIN_MEDS = [
    {"name": "Lisinopril", "dosages": [{"val": 10, "unit": "mg"}, {"val": 20, "unit": "mg"}]},
    {"name": "Metformin", "dosages": [{"val": 500, "unit": "mg"}, {"val": 850, "unit": "mg"}]},
    {"name": "Omeprazole", "dosages": [{"val": 20, "unit": "mg"}, {"val": 40, "unit": "mg"}]},
    {"name": "Atorvastatin", "dosages": [{"val": 10, "unit": "mg"}, {"val": 40, "unit": "mg"}]}
]

# --- POOL 2: HOLD-OUT DATA ---
TEST_CONDITIONS = [
    "Acute Bronchitis", "Insomnia", "Lower Back Pain", "COVID-19", "Fractured Tibia"
]
TEST_MEDS = [
    {"name": "Amoxicillin", "dosages": [{"val": 250, "unit": "mg"}, {"val": 500, "unit": "mg"}]},
    {"name": "Ibuprofen", "dosages": [{"val": 400, "unit": "mg"}, {"val": 600, "unit": "mg"}]},
    {"name": "Paxlovid", "dosages": [{"val": 300, "unit": "mg"}, {"val": 150, "unit": "mg"}]},
    {"name": "Paracetamol", "dosages": [{"val": 500, "unit": "mg"}, {"val": 1000, "unit": "mg"}]}
]

NAMES = ["John Smith", "Jane Doe", "Robert Johnson", "Emily Davis", "Michael Brown", "Sarah Wilson"]
SYMPTOMS = ["coughing", "shortness of breath", "increased thirst", "headache", "heartburn", "fatigue", "pain"]

def generate_sample(is_training=True):
    conditions = TRAIN_CONDITIONS if is_training else TEST_CONDITIONS
    med_pool = TRAIN_MEDS if is_training else TEST_MEDS
    
    name = random.choice(NAMES)
    
    # 1. COMPLEXITY: Missing Data (Null Handling)
    # 20% chance age is NOT mentioned. Model must output null, not hallucinate.
    include_age = random.random() > 0.2
    age = random.randint(18, 90) if include_age else None
    
    condition = random.choice(conditions)
    symptom = random.choice(SYMPTOMS)
    
    # 2. COMPLEXITY: Multi-Medication Support
    num_meds = 2 if random.random() > 0.7 else 1
    selected_meds = random.sample(med_pool, num_meds)
    
    med_descriptions = []
    med_requests_json = []
    
    for med in selected_meds:
        dosage = random.choice(med["dosages"])
        dosage_str = f"{dosage['val']}{dosage['unit']}"
        med_descriptions.append(f"{med['name']} {dosage_str}")
        
        med_requests_json.append({
            "medicationCodeableConcept": {"text": med['name']},
            "dosageInstruction": [
                {
                    "doseAndRate": [
                        {
                            "doseQuantity": {
                                "value": dosage['val'],
                                "unit": dosage['unit']
                            }
                        }
                    ]
                }
            ]
        })
    
    med_text = " and ".join(med_descriptions)
    
    # --- FIX: Ensure Age matches JSON ---
    if include_age:
        age_str_1 = f", {age}y,"
        age_str_2 = f" ({age})"
        age_str_3 = f"{age}yo "
    else:
        age_str_1 = ","
        age_str_2 = "" 
        age_str_3 = ""

    templates = [
        f"Pt {name}{age_str_1} presents w/ {symptom}. Impression: {condition}. Rx: {med_text}.",
        f"Note: {name}{age_str_2} reports {symptom}. Assessment: {condition}. Plan: Start {med_text}.",
        f"History: {age_str_3}{name} c/o {symptom}. PMH: {condition}. Order: {med_text}."
    ]
    note = random.choice(templates)
    
    target_json = {
        "resourceType": "SolarisEMRClinicalSummary", 
        "status": "finished",
        "subject": {
            "reference": f"Patient/{name.replace(' ', '')}",
            "display": name,
            "ageInt": age # Strict Integer OR null
        },
        "reasonCode": [{"text": symptom}],
        "diagnosis": {
            "condition": condition,
            "clinicalStatus": "active"
        },
        "medicationRequest": med_requests_json
    }
    
    return {
        "messages": [
            {"role": "system", "content": "Extract clinical data into strict Solaris-compliant JSON."},
            {"role": "user", "content": note},
            {"role": "assistant", "content": json.dumps(target_json)}
        ]
    }

def main():
    print(f"Generating {NUM_SAMPLES} samples (Fixed Age Logic)...")
    
    train_data = []
    val_data = []
    
    num_train = int(NUM_SAMPLES * 0.9)
    for _ in range(num_train):
        train_data.append(generate_sample(is_training=True))
        
    num_val = NUM_SAMPLES - num_train
    for _ in range(num_val):
        val_data.append(generate_sample(is_training=False))
    
    with open("train.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")
            
    with open("val.jsonl", "w") as f:
        for entry in val_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Done. Run train_lora.py now.")

if __name__ == "__main__":
    main()