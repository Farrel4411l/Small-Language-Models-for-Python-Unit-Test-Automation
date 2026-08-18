import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_lr():
    base_dir = Path(__file__).parent.parent
    state_path = base_dir / "fine_tuning" / "results" / "checkpoint-11880" / "trainer_state.json"
    
    if not state_path.exists():
        print(f"Error: Tidak menemukan {state_path}")
        return
        
    print(f"Membaca {state_path}...")
    with open(state_path, "r", encoding="utf-8") as f:
        state_data = json.load(f)
        
    log_history = state_data.get("log_history", [])
    
    steps = []
    lrs = []
    
    for log in log_history:
        if "learning_rate" in log and "step" in log:
            steps.append(log["step"])
            lrs.append(log["learning_rate"])
            
    if not steps:
        print("Tidak menemukan metrik learning_rate di dalam log history.")
        return
        
    print(f"Ditemukan {len(steps)} titik learning rate.")
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs, label='Learning Rate', color='green', linewidth=2)
    plt.xlabel('Steps (Total 11880 steps = 30 Epochs)')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    out_path = base_dir / "evaluation" / "lr_curve.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grafik LR berhasil disimpan ke: {out_path}")

if __name__ == "__main__":
    plot_lr()
