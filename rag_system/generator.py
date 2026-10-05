import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class RAGGenerator:
    def __init__(self, base_model_id="Qwen/Qwen2.5-7B-Instruct", lora_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[System] Memuat Tokenizer {base_model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        
        print(f"[System] Memuat Base Model {base_model_id} (BF16) ke {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if lora_path and os.path.exists(lora_path):
            print(f"[System] Memuat LoRA Adapter dari {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            
        self.model.eval()
        
    def build_prompt(self, query, chunks):
        # Menyusun konteks dengan format kutipan
        context_str = ""
        for i, chunk in enumerate(chunks, 1):
            reg = chunk['metadata'].get('regulation', 'Unknown')
            pasal = chunk['metadata'].get('pasal', '?')
            content = chunk['content']
            context_str += f"--- DOKUMEN {i} ---\nSumber: {reg} - Pasal {pasal}\n{content}\n\n"
            
        system_prompt = (
            "Anda adalah Asisten Pajak Indonesia yang ahli dan presisi. "
            "Gunakan HANYA dokumen referensi yang diberikan untuk menjawab pertanyaan pengguna. "
            "Jika jawaban memerlukan perhitungan, tunjukkan langkah-langkahnya secara rinci berdasarkan aturan di dokumen. "
            "PENTING: Anda WAJIB mengutip sumber hukum yang Anda gunakan dalam format [Nama Dokumen - Pasal X]."
        )
        
        user_message = f"DOKUMEN REFERENSI:\n{context_str}\n\nPERTANYAAN: {query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text

    def generate(self, query, chunks, max_new_tokens=1024):
        prompt = self.build_prompt(query, chunks)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        
        print("  -> Sedang meng-generate jawaban (Inferensi SLM)...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Temperature sangat rendah untuk factual RAG
                top_p=0.9,
                repetition_penalty=1.05
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

if __name__ == "__main__":
    # Test script jika dijalankan stand-alone
    # Catatan: Ini butuh ~14GB VRAM
    base_dir = os.path.dirname(os.path.dirname(__file__))
    lora_path = os.path.join(base_dir, "fine_tuning", "model_qlora_pph21")
    
    # dummy chunk untuk test
    dummy_chunks = [
        {
            "metadata": {"regulation": "PP 58/2023", "pasal": "1"},
            "content": "Tarif pemotongan PPh Pasal 21 atas penghasilan sehubungan dengan pekerjaan terdiri atas tarif efektif rata-rata (TER) dan tarif Pasal 17 UU PPh."
        }
    ]
    query = "Apa saja tarif pemotongan PPh Pasal 21 menurut PP 58/2023?"
    
    print("Memulai inisialisasi Generator (Bisa OOM jika VRAM tidak cukup)...")
    generator = RAGGenerator(lora_path=lora_path)
    ans = generator.generate(query, dummy_chunks)
    print("\n[JAWABAN]:\n", ans)
