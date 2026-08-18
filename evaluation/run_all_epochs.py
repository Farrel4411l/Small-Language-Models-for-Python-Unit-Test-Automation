import subprocess
import re
import sys
from pathlib import Path

def get_accuracy_from_log(output):
    match = re.search(r"AKURASI MODEL \(PASS@1\) : ([\d\.]+)%", output)
    if match:
        return float(match.group(1))
    return 0.0

def main():
    base_dir = Path(__file__).parent.parent
    checkpoints = {
        5: "checkpoint-1980",
        10: "checkpoint-3960",
        20: "checkpoint-7920",
        25: "checkpoint-9900"
    }
    
    results = {}
    
    print("🚀 Memulai Loop Evaluasi Multi-Epoch OOD...", flush=True)
    for epoch, cp_name in checkpoints.items():
        cp_path = base_dir / "fine_tuning" / "results" / cp_name
        if not cp_path.exists():
            print(f"❌ Checkpoint tidak ditemukan: {cp_path}", flush=True)
            continue
            
        print(f"\n--- Menjalankan Evaluasi OOD untuk Epoch {epoch} ({cp_name}) ---", flush=True)
        
        script_path = base_dir / "evaluation" / "run_metrics.py"
        cmd = ["python", str(script_path), "--split", "ood_test", "--checkpoint", str(cp_path)]
        
        # Menggunakan Popen untuk membaca output baris-per-baris dan menampilkannya secara live (mencegah log kosong)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        full_output = []
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            full_output.append(line)
            
        process.wait()
        
        if process.returncode != 0:
            print(f"❌ Terjadi kesalahan saat mengevaluasi Epoch {epoch}", flush=True)
            continue
            
        acc = get_accuracy_from_log("".join(full_output))
        results[epoch] = acc
        print(f"✅ Epoch {epoch} Selesai! Skor: {acc}%", flush=True)
        
    print("\n" + "="*50, flush=True)
    print("📊 REKAPITULASI AKURASI OOD PER EPOCH", flush=True)
    print("="*50, flush=True)
    for epoch in sorted(results.keys()):
        print(f"Epoch {epoch:02d} : {results[epoch]}%", flush=True)
    print("="*50, flush=True)
    
if __name__ == "__main__":
    main()
