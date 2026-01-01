import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" 
NEW_MODEL_NAME = "Qwen2.5-3B-Clinical-Extract"

# 1. Load Dataset
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token 

# 3. Quantization Config 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, # Strict FP16 computation
)

# 4. Load Base Model
print(f"Loading base model: {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager"  # Force standard attention (disables FlashAttn/SDPA which cause BF16 crashes on the 1070)
)

# --- CONFIG OVERRIDES FOR PASCAL GPUS ---
model.config.torch_dtype = torch.float16 
model.config.use_cache = False 
model = prepare_model_for_kbit_training(model)

# 5. LoRA Configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

# --- DEEP SANITIZATION: REMOVE ALL BFLOAT16 ---
# This function recursively hunts for bfloat16 tensors in parameters AND buffers and converts them to float16 to prevent crashes. TODO: remove now fp16=False is set.
def force_remove_bf16(m):
    count = 0
    for name, param in m.named_parameters():
        count += 1
        if param.dtype == torch.bfloat16:
            print(f"  - Casting Param to fp16: {name}")
            param.data = param.data.to(torch.float16)
    for name, buf in m.named_buffers():
        if buf.dtype == torch.bfloat16:
            print(f"  - Casting Buffer to fp16: {name}")
            buf.data = buf.data.to(torch.float16)
    print(f"******  -> Scanned {count} parameters for bfloat16 artifacts.")

print("******* Scanning model for incompatible bfloat16 types...")
force_remove_bf16(model)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,                     
    per_device_train_batch_size=2, #might need to also set per_device_eval_batch_size=2? Big VRAM jump to 7Gb on 1st validation pass...         
    gradient_accumulation_steps=4,          
    learning_rate=2e-4,                     
    weight_decay=0.001,
    fp16=False,                              # Disable FP16
    bf16=False,                             # Disable BF16
    logging_steps=10,                       
    eval_strategy="steps",                  
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    optim="paged_adamw_32bit",              
    report_to=["none"],
    # Adding gradient checkpointing to help stability.
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False} 
)

# 7. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    args=training_args,
    processing_class=tokenizer,             
)

# 8. Train
print("Starting training...")
trainer.train()

# 9. Save Adapter
print(f"Saving adapter to {NEW_MODEL_NAME}...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)

print("Training complete. Run evaluate_and_plot.py next.")