import matplotlib.pyplot as plt
from pathlib import Path

def plot_accuracy():
    epochs = [5, 10, 15, 20, 25, 30]
    accuracies = [95.2, 91.9, 8.1, 90.3, 77.4, 96.8]
    
    plt.figure(figsize=(10, 6))
    
    # Plot line
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='purple', linewidth=2.5, markersize=8)
    
    # Anotasi titik ekstrem
    plt.annotate('Catastrophic\nCollapse\n(Formatting Error)', xy=(15, 8.1), xytext=(15, 25),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 horizontalalignment='center', color='red', weight='bold')
                 
    plt.annotate('Final Mastery', xy=(30, 96.8), xytext=(28, 80),
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 horizontalalignment='center', color='green', weight='bold')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Akurasi OOD (Pass@1) %', fontsize=12)
    plt.title('Evolusi Akurasi Fungsional Model (Pass@1) vs Epoch', fontsize=14, weight='bold')
    
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    base_dir = Path(__file__).parent.parent
    out_path = base_dir / "evaluation" / "accuracy_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafik Akurasi berhasil disimpan ke: {out_path}")

if __name__ == "__main__":
    plot_accuracy()
