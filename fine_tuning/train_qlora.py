import os
import sqlite3
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 1. KONFIGURASI MODEL
# Qwen2.5-7B-Instruct adalah model 100% gratis, open weights, sangat pintar coding, 
# dan tidak butuh token/lisensi khusus seperti Llama-3!
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

def load_data_from_sqlite():
    """Membaca data dari database SQLite dan mengubahnya ke dataset HuggingFace."""
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / "data" / "processed" / "payroll_tests.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT source_code, generated_test, split_type FROM payroll_qa WHERE split_type IN ('train', 'val')")
    rows = cursor.fetchall()
    conn.close()
    
    # Alpaca Prompt Format
    formatted_data = {"train": {"text": []}, "val": {"text": []}}
    for source_code, generated_test, split_type in rows:
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "Buat unit test menggunakan pytest untuk fungsi Python berikut. Gunakan representasi formula matematika murni di dalam statement assert, JANGAN gunakan hardcoded float.\n\n"
            f"### Input:\n{source_code}\n\n"
            f"### Response:\n{generated_test}"
        )
        if split_type == 'train':
            formatted_data["train"]["text"].append(prompt)
        elif split_type == 'val':
            formatted_data["val"]["text"].append(prompt)
        
    return {
        "train": Dataset.from_dict(formatted_data["train"]),
        "val": Dataset.from_dict(formatted_data["val"])
    }

def main():
    print(f"Mempersiapkan data latih dari SQLite...")
    datasets = load_data_from_sqlite()
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    print(f"Total data latih: {len(train_dataset)} pasang. Data validasi: {len(val_dataset)} pasang.")
    
    print("Memuat Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Mengonfigurasi 4-bit Quantization (QLoRA) agar muat di RTX 4070 8GB...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print(f"Mengunduh dan memuat model raksasa ({MODEL_ID}). Ini mungkin memakan waktu lama...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    print("Menyuntikkan modul LoRA ke dalam model (PEFT)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    # Catatan: Kita tidak lagi memanggil get_peft_model() secara manual di sini.
    # SFTTrainer akan otomatis membungkus model dengan LoRA karena kita mengoper peft_config ke dalamnya.
    
    print("Mempersiapkan Trainer...")
    training_args = SFTConfig(
        output_dir="./fine_tuning/results",
        num_train_epochs=30.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_steps=1,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,                            # RTX 4070 mendukung bfloat16
        max_grad_norm=0.3,
        warmup_steps=5,                       # Mengganti warmup_ratio dengan warmup_steps untuk kompatibilitas
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        eval_strategy="steps",
        eval_steps=50,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )
    
    print("🚀 Memulai proses Fine-Tuning SLM...")
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        
    if last_checkpoint is not None:
        print(f"🔄 Checkpoint ditemukan! Melanjutkan training dari: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    
    print("Menyimpan adapter LoRA hasil latihan...")
    output_model_path = "./fine_tuning/model_qlora_pph21"
    trainer.model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    print(f"🎉 Selesai! Model spesialis PPh 21 Anda telah tersimpan di: {output_model_path}")

if __name__ == "__main__":
    main()
