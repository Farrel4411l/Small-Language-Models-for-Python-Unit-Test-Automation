import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_triad_accuracy():
    base_dir = Path(__file__).parent.parent
    results_file = base_dir / "evaluation" / "triad_results.json"
    
    if not results_file.exists():
        print("Data belum tersedia.")
        return
        
    with open(results_file, "r") as f:
        results = json.load(f)
        
    epochs = []
    math_acc = []
    ling_acc = []
    mix_acc = []
    
    for epoch in range(1, 31):
        epoch_str = str(epoch)
        if epoch_str in results:
            epochs.append(epoch)
            math_acc.append(results[epoch_str].get('ood_math', 0))
            ling_acc.append(results[epoch_str].get('ood_linguistic', 0))
            mix_acc.append(results[epoch_str].get('ood_mixed', 0))
            
    plt.figure(figsize=(12, 7))
    
    plt.plot(epochs, math_acc, marker='o', linestyle='-', color='blue', linewidth=2, label='OOD Math (Extremes)')
    plt.plot(epochs, ling_acc, marker='s', linestyle='-', color='green', linewidth=2, label='OOD Linguistic (Slang/JSON)')
    plt.plot(epochs, mix_acc, marker='^', linestyle='-', color='red', linewidth=2, label='OOD Mixed (Extreme + Slang)')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Akurasi OOD (Pass@1) %', fontsize=12)
    plt.title('Evolusi "Otak SLM": Logika Matematika vs Pemahaman Linguistik', fontsize=14, weight='bold')
    
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    
    out_path = base_dir / "evaluation" / "triad_accuracy_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafik Triad Akurasi berhasil disimpan ke: {out_path}")

if __name__ == "__main__":
    plot_triad_accuracy()
