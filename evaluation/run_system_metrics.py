import sys
import time
import torch
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))

from evaluation.inference import ModelInferencer


def profile_system_metrics(inferencer, prompts):
    print("=" * 50)
    print("📊 SYSTEM METRICS PROFILING (VRAM & THROUGHPUT)")
    print("=" * 50)

    # Reset CUDA memory for accurate profiling
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    total_tokens_generated = 0
    total_time = 0.0

    print(f"Running stress test across {len(prompts)} samples...")

    for i, prompt in enumerate(prompts):
        start_time = time.time()

        # Execute generation
        formatted_prompt = (
            f"Lengkapi fungsi Python berikut:\n\n```python\n{prompt}\n```"
        )
        response = inferencer.generate_general(formatted_prompt)

        end_time = time.time()

        # Calculate number of generated tokens using tokenizer
        tokens_count = len(inferencer.tokenizer.encode(response))

        total_tokens_generated += tokens_count
        total_time += end_time - start_time

    # Calculate Metrics
    throughput = total_tokens_generated / total_time
    max_vram_bytes = torch.cuda.max_memory_allocated()
    max_vram_gb = max_vram_bytes / (1024**3)

    print(f"\n[PROFILING RESULTS]")
    print(f"⏱️ Total Time         : {total_time:.2f} seconds")
    print(f"🪙 Total Tokens       : {total_tokens_generated} tokens")
    print(f"🚀 Inference Speed    : {throughput:.2f} tokens/second")
    print(f"💾 Peak VRAM Usage    : {max_vram_gb:.2f} GB")
    print("=" * 50)

    return throughput, max_vram_gb


def main():
    print("Loading Model and LoRA for System Profiling...")
    checkpoint_path = base_dir / "fine_tuning" / "results_guardrails" / "final_model"
    inferencer = ModelInferencer(lora_path=str(checkpoint_path))

    # Dummy prompts collection simulating long code queries
    dummy_prompts = [
        "def hitung_pph21_karyawan(gaji_pokok, tunjangan, ptkp_status):",
        "def kalkulasi_pajak_tahunan(bruto, pengurang, npwp_valid):",
        "def validasi_format_npwp(npwp_string):",
        "def get_tarif_efektif_rata_rata(kategori, pkp):",
        "def hitung_thr_dan_bonus_kena_pajak(gaji, thr, bonus):",
    ]

    profile_system_metrics(inferencer, dummy_prompts)


if __name__ == "__main__":
    main()
