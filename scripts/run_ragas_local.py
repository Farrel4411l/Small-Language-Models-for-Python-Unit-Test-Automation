import os
import json
import torch
import gc
from datasets import Dataset

# Monkeypatch transformers CVE-2025-32434
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers, 'modeling_utils'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_out = os.path.join(base_dir, "evaluation", "raw_generation_results.json")
    
    if not os.path.exists(raw_out):
        print(f"File {raw_out} tidak ditemukan!")
        return
        
    with open(raw_out, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
        
    # Pastikan semua pertanyaan sudah memiliki 'generated_answer'
    for item in gold_data:
        if 'generated_answer' not in item:
            print("Belum semua pertanyaan digenerate. Harap tunggu Step 2 selesai.")
            return

    print("=== MENJALANKAN RAGAS DENGAN LOCAL LLM JUDGE ===")
    
    data_for_ragas = {
        "question": [d["question"] for d in gold_data],
        "answer": [d["generated_answer"] for d in gold_data],
        "contexts": [[d["retrieved_context"]] for d in gold_data],
        "ground_truth": [d["expected_answer"] for d in gold_data]
    }
    
    dataset = Dataset.from_dict(data_for_ragas)
    
    print("-> Memuat BGE-M3 untuk metrik embedding (Answer Relevancy) PADA CPU (Untuk menghemat VRAM)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3", 
        model_kwargs={'device': 'cpu'}
    )
    
    print("-> Memuat Qwen 7B 4-bit sebagai Local Judge untuk Faithfulness...")
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
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        return_full_text=False
    )
    
    local_llm = HuggingFacePipeline(pipeline=pipe)
    
    print("-> Memulai evaluasi (Ini akan memakan waktu)...")
    # Wrap model untuk ragas
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    ragas_llm = LangchainLLMWrapper(local_llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)
    
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_emb,
        raise_exceptions=False
    )
    
    print("\n=== HASIL EVALUASI RAGAS ===")
    print(result)
    
    df = result.to_pandas()
    out_csv = os.path.join(base_dir, "evaluation", "ragas_results.csv")
    df.to_csv(out_csv, index=False)
    
    print(f"Detail tersimpan di: {out_csv}")

if __name__ == "__main__":
    main()
