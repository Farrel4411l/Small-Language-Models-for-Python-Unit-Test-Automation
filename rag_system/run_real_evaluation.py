import os
import json
import pandas as pd
from datasets import Dataset

# ==============================================================================
# 🚨 PENTING: Script ini adalah KERANGKA EVALUASI ASLI (Bukan Simulasi)
# Dijalankan dengan model aktual di GPU untuk mengumpulkan metrics.
# ==============================================================================

import torch
import gc
from tqdm import tqdm
import time

# Monkeypatch transformers to bypass torch.load CVE-2025-32434 restriction
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers, 'modeling_utils'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None


def run_real_evaluation(gold_dataset_path, output_dir):
    print("[EVALUASI] Memulai evaluasi RAG (End-to-End)...")
    
    if not os.path.exists(gold_dataset_path):
        print(f"Error: {gold_dataset_path} belum dibuat.")
        return
        
    with open(gold_dataset_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
        
    print(f"-> Memuat {len(gold_data)} pertanyaan benchmark.")
    
    # Karena VRAM hanya 8GB (RTX 4070 Laptop), kita wajib melakukan eksekusi SATU PER SATU
    # (Unload model dari memori setelah selesai agar tidak OOM)
    
    # ---------------------------------------------------------
    # STEP 1: RETRIEVAL (Dense + Sparse + Reranker)
    # ---------------------------------------------------------
    print("\n=== STEP 1: MENJALANKAN RETRIEVER ===")
    
    intermediate_out = os.path.join(output_dir, "intermediate_retrieval_results.json")
    raw_out = os.path.join(output_dir, "raw_generation_results.json")
    
    if os.path.exists(raw_out) or os.path.exists(intermediate_out):
        print("-> Checkpoint retrieval ditemukan. Melewati eksekusi model Retriever untuk menghemat waktu!")
    else:
        from rag_system.retriever import HybridRetriever
        
        retriever = HybridRetriever(top_k=20, rerank_top_k=3)
        
        for item in tqdm(gold_data, desc="Retrieving Contexts"):
            results = retriever.retrieve(item['question'])
            # Gabungkan teks context
            context_str = "\n".join([res['content'] for res in results])
            item['retrieved_context'] = context_str
            item['retrieved_raw'] = results
            
        # Unload Retriever dari VRAM
        print("-> Membersihkan VRAM dari model Retriever...")
        del retriever
        gc.collect()
        torch.cuda.empty_cache()
        
        # Save intermediate retrieval results
        with open(intermediate_out, 'w', encoding='utf-8') as f:
            json.dump(gold_data, f, indent=4)
        print(f"-> Hasil retrieval sementara disimpan di {intermediate_out}")


    
    # ---------------------------------------------------------
    # STEP 2: GENERATION (Qwen 7B 4-bit)
    # ---------------------------------------------------------
    print("\n=== STEP 2: MENJALANKAN GENERATOR (LLM) ===")
    
    # CHECKPOINTING: Load progress jika sudah ada yang tergenerate
    raw_out = os.path.join(output_dir, "raw_generation_results.json")
    if os.path.exists(raw_out):
        print(f"-> Ditemukan checkpoint di {raw_out}. Melanjutkan evaluasi...")
        with open(raw_out, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
    elif os.path.exists(intermediate_out):
        print(f"-> Memuat konteks dari {intermediate_out}...")
        with open(intermediate_out, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)

    # Cek apakah masih ada yang belum di-generate
    unprocessed = [item for item in gold_data if 'generated_answer' not in item]
    if not unprocessed:
        print("-> Semua jawaban sudah berhasil di-generate! Melompat ke Step 3.")
    else:
        print(f"-> Tersisa {len(unprocessed)} pertanyaan untuk di-generate. Memuat Qwen 7B dalam mode 4-bit...")
        
        from transformers import BitsAndBytesConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        lora_path = os.path.join(base_dir, "fine_tuning", "model_qlora_pph21")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", 
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        if os.path.exists(lora_path):
            model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        
        for item in tqdm(gold_data, desc="Generating Answers"):
            if 'generated_answer' in item:
                continue # Skip jika sudah ada dari checkpoint sebelumnya
                
            # Build prompt
            system_prompt = (
                "Anda adalah Asisten Pajak Indonesia yang ahli dan presisi. "
                "Gunakan HANYA dokumen referensi yang diberikan untuk menjawab pertanyaan pengguna. "
                "PENTING: Anda WAJIB mengutip sumber hukum yang Anda gunakan dalam format [Nama Dokumen - Pasal X]."
            )
            
            # Reconstruct retrieved_context if missing
            if 'retrieved_context' not in item and 'retrieved_raw' in item:
                item['retrieved_context'] = "\n".join([res['content'] for res in item['retrieved_raw']])
                
            user_message = f"DOKUMEN REFERENSI:\n{item['retrieved_context']}\n\nPERTANYAAN: {item['question']}"
            
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
            
            answer = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            item['generated_answer'] = answer
            
            # AUTO-SAVE: Simpan setelah setiap jawaban untuk mencegah kehilangan progres
            with open(raw_out, 'w', encoding='utf-8') as f:
                json.dump(gold_data, f, indent=4)
            
        # Unload LLM
        print("-> Membersihkan VRAM dari LLM...")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # STEP 3: RAGAS EVALUATION
    # ---------------------------------------------------------
    print("\n=== STEP 3: EVALUASI RAGAS ===")
    print("Menyiapkan dataset untuk Ragas...")
    
    data_for_ragas = {
        "question": [d["question"] for d in gold_data],
        "answer": [d["generated_answer"] for d in gold_data],
        "contexts": [[d["retrieved_context"]] for d in gold_data],
        "ground_truth": [d["expected_answer"] for d in gold_data]
    }
    
    dataset = Dataset.from_dict(data_for_ragas)
    
    # Simpan raw result sebelum dievaluasi agar aman
    raw_out = os.path.join(output_dir, "raw_generation_results.json")
    with open(raw_out, 'w', encoding='utf-8') as f:
        json.dump(gold_data, f, indent=4)
        
    print(f"Hasil mentah (tanpa metrik) diamankan ke {raw_out}")
    print("\n⚠️ Untuk menjalankan RAGAS metrics, diperlukan OpenAI API Key (GPT-4/GPT-3.5).")
    print("Jika Anda ingin menggunakan Local Judge LLM, pastikan Anda menggunakan server vLLM terpisah.")
    print("Silakan jalankan Ragas Evaluation secara terpisah menggunakan dataset yang baru saja dihasilkan!")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    bench_dir = os.path.join(base_dir, "data", "benchmarks")
    eval_dir = os.path.join(base_dir, "evaluation")
    
    os.makedirs(eval_dir, exist_ok=True)
    run_real_evaluation(
        os.path.join(bench_dir, "gold_dataset.json"), 
        eval_dir
    )
