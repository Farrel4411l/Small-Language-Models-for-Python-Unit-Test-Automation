import sys
from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))

from evaluation.inference import ModelInferencer

def main():
    print("==================================================")
    print("🕵️ UJI BLIND TEST SESUNGGUHNYA (THE REAL OOD)")
    print("==================================================")
    
    real_prompt = """
    Bang, gaji bulanan gue 7.5 juta. Status gue belum nikah dan gak ada tanggungan sama sekali (TK/0). Tolong buatin script python untuk ngitung potongan PPh21 bulanan gue pakai aturan TER (Tarif Efektif Rata-rata) terbaru dong. Jangan lupa sekalian bikinin 3 test case pytest pakai gaji 4 juta, 7.5 juta, dan 15 juta dengan status yang sama.
    """
    
    print("\n--- MENGUJI FINE-TUNED MODEL (PHASE 2) ---")
    lora_path = str(base_dir / "fine_tuning" / "results_guardrails" / "final_model")
    try:
        inferencer = ModelInferencer(lora_path=lora_path)
        
        # Kita gunakan generate_general karena ini bukan prompt murni dari dataset
        response = inferencer.generate_general(real_prompt)
        print("\n[JAWABAN FINE-TUNED MODEL]:")
        print(response)
        del inferencer
        import torch
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error pada Fine-Tuned: {e}")

if __name__ == "__main__":
    main()
