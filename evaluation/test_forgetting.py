import sys
from pathlib import Path

# Add project root to sys.path
base_dir = Path(__file__).parent.parent
sys.path.append(str(base_dir))

from inference import ModelInferencer

def main():
    print("="*60)
    print("🧠 UJI CATASTROPHIC FORGETTING (INGATAN DASAR)")
    print("="*60)
    
    # Load model pada checkpoint terakhir (Epoch 30)
    checkpoint_path = base_dir / "fine_tuning" / "results_guardrails" / "final_model"
    inferencer = ModelInferencer(lora_path=str(checkpoint_path))
    
    test_cases = [
        {
            "name": "Sapaan / Chit-chat",
            "prompt": "Halo AI, siapa namamu dan apa tujuanmu diciptakan?"
        },
        {
            "name": "Python Dasar - Fibonacci",
            "prompt": "Tolong tuliskan fungsi Python sederhana untuk mencetak deret Fibonacci hingga angka N."
        },
        {
            "name": "Python Data Science",
            "prompt": "Bagaimana cara membaca file CSV bernama 'data.csv' menggunakan library pandas di Python?"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- [Ujian {i}/3] Domain: {test['name']} ---")
        print(f"Prompt: {test['prompt']}")
        print("Menunggu jawaban SLM...")
        
        # Override sistem prompt default karena inferencer.py menggunakan prompt pajak
        # Kita akan modifikasi format stringnya langsung di sini untuk uji coba
        # Tapi karena `generate_test` di inference.py keras kepala menambah "Berikan kode pytest..."
        # Kita panggil saja metode raw model generate.
        
        # Build prompt format Llama3/Qwen
        raw_prompt = f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{test['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = inferencer.tokenizer(raw_prompt, return_tensors="pt").to(inferencer.model.device)
        outputs = inferencer.model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=inferencer.tokenizer.eos_token_id
        )
        
        response = inferencer.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"\nJawaban SLM:\n{response}")
        print("-" * 50)
        
if __name__ == "__main__":
    main()
