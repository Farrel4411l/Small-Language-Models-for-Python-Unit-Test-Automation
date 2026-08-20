import sys
from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))

from evaluation.inference import ModelInferencer

def main():
    print("==================================================")
    print("🕵️ UJI BLIND TEST (REAL-WORLD / OUT-OF-DISTRIBUTION)")
    print("==================================================")
    
    # Prompt bergaya forum diskusi/konsultasi (tanpa template rapi)
    messy_prompt = """
    Bang, tolong bantu buatin kode python sama unit testnya dong buat ngitung pajak penghasilan. 
    Aturannya gini: kalau gaji dia di bawah 5 juta sebulan (alias 60 juta setahun), dia bebas pajak (PTKP). 
    Tapi kalau lebih dari itu, sisa dari gajinya (setelah dikurangi 60 juta) dipotong pajak 5%.
    
    Tolong buatin fungsi `hitung_pajak_bang(gaji_setahun)` dan sekalian bikinin 3 test case pake pytest ya bang.
    Gaji setahunnya: 50 juta, 60 juta, dan 100 juta.
    """
    
    print("\n--- MENGUJI FINE-TUNED MODEL (PHASE 2) ---")
    lora_path = str(base_dir / "fine_tuning" / "results_guardrails" / "final_model")
    try:
        inferencer = ModelInferencer(lora_path=lora_path)
        
        # Kita gunakan generate_general karena ini bukan prompt murni dari dataset
        response = inferencer.generate_general(messy_prompt)
        print("\n[JAWABAN FINE-TUNED MODEL]:")
        print(response)
        del inferencer
        import torch
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error pada Fine-Tuned: {e}")

if __name__ == "__main__":
    main()
