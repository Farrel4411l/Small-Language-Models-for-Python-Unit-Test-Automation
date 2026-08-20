import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Configure Gemini API using the official SDK (google-genai)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    # Use gemini-flash-latest model to avoid deprecated model endpoints
    MODEL_ID = "gemini-flash-latest"
else:
    print("PERINGATAN: GEMINI_API_KEY tidak ditemukan di .env!")
    exit(1)


def generate_pytest_with_llm(source_code, max_retries=3):
    """
    Sends a prompt to the teacher LLM to generate pytest unit tests
    focused on Indonesian PPh 21 business rule compliance.
    Includes automated retry logic for rate limit handling.
    """
    prompt = f"""
    Kamu adalah seorang Senior QA Engineer dan ahli perpajakan PPh 21 Indonesia.
    Tugasmu adalah membuat unit test menggunakan `pytest` untuk source code Python berikut.
    
    ATURAN BISNIS YANG HARUS DIVALIDASI:
    1. Pastikan logika pemotongan pajak progresif (Pasal 17) benar.
    2. Pastikan logika Penghasilan Tidak Kena Pajak (PTKP) sesuai dengan status (TK/0, K/1, dll).
    3. Tes harus mencakup edge cases (misal: gaji di bawah PTKP sehingga pajak 0, gaji sangat besar).
    
    SOURCE CODE:
    ```python
    {source_code}
    ```
    
    INSTRUKSI OUTPUT:
    Berikan HANYA kode Python (script pytest) yang menguji fungsi di atas. 
    Jangan berikan penjelasan apapun. Pastikan kodenya bisa langsung dieksekusi.
    """

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            # Clean markdown code fences (```python ... ```) from output
            clean_text = (
                response.text.replace("```python", "").replace("```", "").strip()
            )
            return clean_text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(
                    f"Terkena limit API (429). Menunggu 30 detik untuk retry... (Percobaan {attempt+1}/{max_retries})"
                )
                time.sleep(30)
            else:
                print(f"Error fatal saat memanggil LLM: {e}")
                return None

    print("Gagal setelah beberapa kali percobaan retry.")
    return None


def main():
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "data" / "processed" / "cleaned_payroll_data.json"
    output_file = base_dir / "data" / "processed" / "qa_dataset.json"

    if not input_file.exists():
        print("Data bersih tidak ditemukan. Jalankan data_cleaner.py terlebih dahulu.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        cleaned_data = json.load(f)

    # Check if existing processed data is available
    dataset = []
    existing_files = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            existing_files = {item["file_name"] for item in dataset}

    print(f"Melanjutkan pembuatan Unit Test... (Sudah ada {len(existing_files)} file)")

    # Process all files in cleaned_data
    for i, item in enumerate(cleaned_data):
        # Skip previously generated files
        if item["file_name"] in existing_files:
            continue

        print(
            f"[{i+1}/{len(cleaned_data)}] Men-generate test untuk file: {item['file_name']}..."
        )

        generated_test = generate_pytest_with_llm(item["clean_code"])

        if generated_test:
            dataset.append(
                {
                    "repo": item["repo"],
                    "file_name": item["file_name"],
                    "source_code": item["clean_code"],
                    "generated_test": generated_test,
                }
            )

            # Incrementally save progress to prevent data loss on interruption
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)

        # Standard delay to avoid API rate limiting
        time.sleep(4)

    print(f"Selesai! Total ada {len(dataset)} pasang data latih yang terkumpul.")
    print(f"Dataset disimpan di: {output_file}")


if __name__ == "__main__":
    main()
