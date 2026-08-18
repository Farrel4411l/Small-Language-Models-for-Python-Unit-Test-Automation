import subprocess
import re
import sys
import json
import os
from pathlib import Path

def get_accuracy_from_log(output):
    match = re.search(r"AKURASI MODEL \(PASS@1\) : ([\d\.]+)%", output)
    if match:
        return float(match.group(1))
    return 0.0

def main():
    base_dir = Path(__file__).parent.parent
    results_file = base_dir / "evaluation" / "triad_results.json"
    
    # Checkpoints dari Epoch 1 (396) s/d Epoch 30 (11880)
    checkpoints = {i: f"checkpoint-{i * 396}" for i in range(1, 31)}
    
    splits = ["ood_math", "ood_linguistic", "ood_mixed"]
    
    # Load existing results if any
    if results_file.exists():
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        results = {}
        
    print("🚀 Memulai Loop Evaluasi TRIAD OOD (30 Epochs)...", flush=True)
    
    for epoch in range(1, 31):
        epoch_str = str(epoch)
        if epoch_str not in results:
            results[epoch_str] = {}
            
        cp_name = checkpoints[epoch]
        cp_path = base_dir / "fine_tuning" / "results" / cp_name
        
        if not cp_path.exists():
            print(f"❌ Checkpoint tidak ditemukan: {cp_path}", flush=True)
            continue
            
        print(f"\n=============================================", flush=True)
        print(f"🏁 MEMULAI EPOCH {epoch} ({cp_name})", flush=True)
        print(f"=============================================", flush=True)
        
        for split in splits:
            if split in results[epoch_str]:
                print(f"⏭️  Melewati {split} untuk Epoch {epoch} (Sudah ada di JSON: {results[epoch_str][split]}%)", flush=True)
                continue
                
            print(f"\n--- Menjalankan Split: {split.upper()} ---", flush=True)
            
            script_path = base_dir / "evaluation" / "run_metrics.py"
            cmd = ["python", str(script_path), "--split", split, "--checkpoint", str(cp_path)]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            
            full_output = []
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                full_output.append(line)
                
            process.wait()
            
            if process.returncode != 0:
                print(f"❌ Terjadi kesalahan saat mengevaluasi Epoch {epoch} - {split}", flush=True)
                continue
                
            acc = get_accuracy_from_log("".join(full_output))
            results[epoch_str][split] = acc
            print(f"✅ Selesai {split}! Skor: {acc}%", flush=True)
            
            # Save incrementally
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)
                
    print("\n" + "="*50, flush=True)
    print("📊 REKAPITULASI AKURASI TRIAD OOD (30 EPOCH)", flush=True)
    print("="*50, flush=True)
    for epoch in range(1, 31):
        epoch_str = str(epoch)
        if epoch_str in results:
            r = results[epoch_str]
            m = r.get('ood_math', '-')
            l = r.get('ood_linguistic', '-')
            x = r.get('ood_mixed', '-')
            print(f"Epoch {epoch:02d} | Math: {m}% | Ling: {l}% | Mix: {x}%", flush=True)
    print("="*50, flush=True)
    
if __name__ == "__main__":
    main()
