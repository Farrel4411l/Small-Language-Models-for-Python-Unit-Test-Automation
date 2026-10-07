import os
import json
import gc
import torch
import time

# Monkeypatch transformers
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers, 'modeling_utils'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from rag_system.retriever import HybridRetriever

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    bench_dir = os.path.join(base_dir, "data", "benchmarks")
    eval_dir = os.path.join(base_dir, "evaluation")
    gold_dataset_path = os.path.join(bench_dir, "gold_dataset.json")
    
    with open(gold_dataset_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
        
    # Hanya sample 10 aja untuk cepat kalau butuh, tapi karena cepat kita run semua 50
    retriever = HybridRetriever(top_k=20, rerank_top_k=3)
    
    for item in gold_data:
        results = retriever.retrieve(item['question'])
        item['retrieved_raw'] = results
        
    out_path = os.path.join(eval_dir, "intermediate_retrieval_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(gold_data, f, indent=4)
        
    print(f"Selesai! Disimpan ke {out_path}")

if __name__ == "__main__":
    main()
