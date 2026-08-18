import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

def main():
    print("="*60)
    print("🚀 MEMULAI INCREMENTAL FINE-TUNING (DATA BLENDING)")
    print("="*60)
    
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    lora_checkpoint = "./fine_tuning/results/checkpoint-11880"
    dataset_path = "./data/processed/payroll_guardrails.jsonl"
    output_dir = "./fine_tuning/results_guardrails"
    
    # 1. Load Dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    def format_prompt(examples):
        prompts = []
        for instruction, output in zip(examples['instruction'], examples['output']):
            prompt = f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            prompts.append(prompt)
        return {"text": prompts}
        
    dataset = dataset.map(format_prompt, batched=True)
    
    # 2. Setup Quantization (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 3. Load Base Model
    print("Memuat Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # 4. Load LoRA Adapter in Trainable Mode
    print("Memuat LoRA Checkpoint (Epoch 30)...")
    model = PeftModel.from_pretrained(base_model, lora_checkpoint, is_trainable=True)
    
    # 5. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 6. Training Arguments for 3 Epochs
    # Karena datanya cuma 200, 3 epoch akan sangat cepat (sekitar 15 menit)
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-5, # LR lebih kecil karena ini fine-tuning lanjutan
        logging_steps=5,
        save_strategy="epoch", # Save setiap selesai 1 epoch
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=True,
        dataset_text_field="text"
    )
    
    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )
    
    print("Memulai Training Incremental...")
    trainer.train()
    
    # 8. Save Final Model
    final_path = os.path.join(output_dir, "final_model")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Incremental Fine-Tuning selesai! Disimpan di {final_path}")

if __name__ == "__main__":
    main()
