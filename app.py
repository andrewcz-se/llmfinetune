from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
from groq import Groq  

app = Flask(__name__)

# --- Configuration ---
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "Qwen2.5-3B-Clinical-Extract"

# Global model variables
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    print("Loading model into VRAM... this might take a minute.")
    
    # 1. 4-Bit Config 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 2. Load Base Model 
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager", # Critical for GTX 1070
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # 3. Attach Adapter 
    # Load the PeftModel on top. 
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Model loaded successfully!")

# Load on startup
try:
    load_model()
except Exception as e:
    print(f"Warning: Could not load local LLM models. Inference will fail. Error: {e}")

def run_inference(prompt, use_adapter=True):
    # Prompt Template (No Schema Definition - Blind Test)
    messages = [
        {"role": "system", "content": "Extract clinical data into strict Solaris-compliant JSON."},
        {"role": "user", "content": prompt}
    ]
    
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    # Inference Logic
    with torch.no_grad():
        if use_adapter:
            # Normal run (Adapter Active)
            generated_ids = model.generate(**inputs, max_new_tokens=400, temperature=0.1)
        else:
            # Context Manager: Temporarily detach adapter to behave like Base Model
            with model.disable_adapter():
                generated_ids = model.generate(**inputs, max_new_tokens=400, temperature=0.1)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/compare', methods=['POST'])
def compare():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    # Run Base Model
    base_output = run_inference(prompt, use_adapter=False)
    
    # Run Tuned Model
    tuned_output = run_inference(prompt, use_adapter=True)
    
    return jsonify({
        "base": base_output,
        "tuned": tuned_output
    })

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Ensure API Key is present
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return jsonify({'error': 'GROQ_API_KEY not set in server environment'}), 500

    try:
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Send to Groq Whisper
        transcription = client.audio.transcriptions.create(
            file=(audio_file.filename, audio_file.read()), 
            model="whisper-large-v3",
            response_format="json",
            language="en",
            temperature=0.0
        )
        
        return jsonify({'text': transcription.text})
        
    except Exception as e:
        print(f"Transcription Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)