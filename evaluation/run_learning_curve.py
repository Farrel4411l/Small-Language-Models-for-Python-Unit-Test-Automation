import sys
import os
import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))


def extract_step(checkpoint_name):
    match = re.search(r"checkpoint-(\d+)", checkpoint_name)
    if match:
        return int(match.group(1))
    return 0


def run_evaluation(checkpoint_path):
    print(f"\n--- Evaluating {checkpoint_path} ---", flush=True)

    cmd = [
        "python",
        "-u",
        str(base_dir / "evaluation" / "run_metrics.py"),
        "--split",
        "test",
        "--checkpoint",
        str(checkpoint_path),
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    full_output = []
    for line in process.stdout:
        print(line, end="", flush=True)
        full_output.append(line)

    process.wait()
    output = "".join(full_output)

    # Search for accuracy metric in output
    match = re.search(
        r"🏆 (?:AKURASI MODEL|MODEL ACCURACY) \(PASS@1\) : ([\d\.]+)%", output
    )
    if not match:
        match = re.search(r"PASS@1.*?([\d\.]+)%", output)
    if match:
        acc = float(match.group(1))
        print(f"Accuracy: {acc:.1f}%", flush=True)
        return acc
    else:
        print("Failed to find accuracy score. Output:")
        print(output)
        return 0.0


def main():
    print("==================================================")
    print("📈 STARTING LEARNING CURVE EVALUATION (SUBPROCESS MODE)")
    print("==================================================")

    checkpoints_dir = base_dir / "fine_tuning" / "results"

    # Retrieve all checkpoint folders and sort by step
    all_checkpoints = [
        d for d in checkpoints_dir.iterdir() if d.is_dir() and "checkpoint" in d.name
    ]
    all_checkpoints = sorted(all_checkpoints, key=lambda x: extract_step(x.name))

    if len(all_checkpoints) > 10:
        indices = np.linspace(0, len(all_checkpoints) - 1, 10, dtype=int)
        selected_checkpoints = [all_checkpoints[i] for i in indices]
    else:
        selected_checkpoints = all_checkpoints

    steps = []
    accuracies = []

    # Base Model (Step 0)
    acc = run_evaluation("NONE")
    steps.append(0)
    accuracies.append(acc)

    # Loop over selected checkpoints
    for cp in selected_checkpoints:
        step = extract_step(cp.name)
        acc = run_evaluation(cp)
        steps.append(step)
        accuracies.append(acc)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(steps, accuracies, marker="o", linestyle="-", color="b", linewidth=2)
    plt.title("Learning Curve: Model Accuracy on PPh21 Test Set vs Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Accuracy (Pass@1) %")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim(-5, 105)

    for i, txt in enumerate(accuracies):
        plt.annotate(
            f"{txt:.1f}%",
            (steps[i], accuracies[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plot_path = base_dir / "learning_curve_test.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Plot successfully saved to: {plot_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
