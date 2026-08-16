import json
import matplotlib.pyplot as plt
import os
from pathlib import Path

def plot_loss(trainer_state_path, output_path):
    print(f"Membaca {trainer_state_path}...")
    with open(trainer_state_path, "r") as f:
        data = json.load(f)
    
    log_history = data.get("log_history", [])
    
    train_epochs = []
    train_loss = []
    
    eval_epochs = []
    eval_loss = []
    
    for entry in log_history:
        # Pengecekan training loss
        if "loss" in entry and "epoch" in entry:
            train_epochs.append(entry["epoch"])
            train_loss.append(entry["loss"])
            
        # Pengecekan validation (eval) loss
        if "eval_loss" in entry and "epoch" in entry:
            eval_epochs.append(entry["epoch"])
            eval_loss.append(entry["eval_loss"])
            
    if not train_loss:
        print("Data training loss tidak ditemukan.")
        return
        
    print(f"Ditemukan {len(train_loss)} titik training loss dan {len(eval_loss)} titik eval loss.")
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_loss, label="Training Loss", color="blue", alpha=0.7)
    
    if eval_loss:
        plt.plot(eval_epochs, eval_loss, label="Validation Loss", color="orange", linewidth=2)
        
    plt.title("Training vs Validation Loss (Anti-Overfitting Analysis)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Grafik loss berhasil disimpan ke: {output_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    trainer_state = base_dir / "fine_tuning" / "results" / "checkpoint-11880" / "trainer_state.json"
    output_png = base_dir / "evaluation" / "loss_curve.png"
    
    if trainer_state.exists():
        plot_loss(trainer_state, output_png)
    else:
        print(f"Error: {trainer_state} tidak ditemukan.")
