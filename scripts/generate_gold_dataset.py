import os
import json

def generate_gold_skeleton():
    """
    Men-generate 50 struktur pertanyaan kosong untuk diisi secara manual.
    Sebaiknya diisi oleh tax consultant / ahli pajak untuk Ground Truth (Fase 4).
    """
    dataset = []
    for i in range(1, 51):
        dataset.append({
            "id": f"q_{i:03d}",
            "question": "",
            "expected_answer": "",
            "ground_truth_context": []
        })
        
    base_dir = os.path.dirname(os.path.dirname(__file__))
    bench_dir = os.path.join(base_dir, "data", "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)
    
    out_path = os.path.join(bench_dir, "gold_dataset_50_skeleton.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Skeleton 50 Gold Dataset berhasil di-generate di: {out_path}")
    print("SILAKAN ISI SECARA MANUAL MENGGUNAKAN SUMBER HUKUM ASLI (UU, PMK, PP).")

if __name__ == "__main__":
    generate_gold_skeleton()
