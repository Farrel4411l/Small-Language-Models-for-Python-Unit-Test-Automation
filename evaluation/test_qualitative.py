import argparse
from inference import ModelInferencer

def test_qualitative(checkpoint):
    print(f"=====================================")
    print(f"🧠 MENGUJI KUALITAS LINGUISTIK MODEL")
    print(f"📦 Checkpoint: {checkpoint}")
    print(f"=====================================\n")
    
    inferencer = ModelInferencer(lora_path=checkpoint)
    
    # Prompt Slang / Tidak Formal
    prompt = """# bang, tolongin buatin fungsi ngitung pph dong buat gaji.
# ptkpnya 54jt yak. klo dibawah nol kaga usah bayar.
# potongannya 15% aja. thx
def ngitung_pajak(gaji_kotor):
    ptkp = 54000000
    sisa = gaji_kotor - ptkp
    if sisa <= 0: return 0
    return sisa * 0.15"""

    print("📜 Prompt yang diberikan (Slang):")
    print(prompt)
    print("\n" + "-"*40 + "\n")
    
    print("🤖 Menunggu respon AI...")
    response = inferencer.generate_test(prompt)
    
    print("📝 Jawaban Model:")
    print(response)
    print("\n=====================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path ke checkpoint model")
    args = parser.parse_args()
    
    test_qualitative(args.checkpoint)
