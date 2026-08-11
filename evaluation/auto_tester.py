import re
import subprocess
import os

class AutoTester:
    def __init__(self, temp_filename="test_temp_eval.py"):
        self.temp_filename = temp_filename

    def _extract_code(self, model_response):
        """Mengekstrak blok kode Python dari respons markdown LLM."""
        # Mencari blok kode yang diawali dengan ```python dan diakhiri dengan ```
        pattern = r"```python(.*?)```"
        match = re.search(pattern, model_response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
            
        # Jika model terpotong di tengah jalan sehingga tidak ada backtick penutup
        pattern_incomplete = r"```python(.*)"
        match_incomplete = re.search(pattern_incomplete, model_response, re.DOTALL)
        if match_incomplete:
            return match_incomplete.group(1).strip()
        
        # Fallback: jika model lupa menulis 'python', coba ambil blok kode biasa
        pattern_fallback = r"```(.*?)```"
        match_fallback = re.search(pattern_fallback, model_response, re.DOTALL)
        if match_fallback:
            return match_fallback.group(1).strip()
            
        # Jika sama sekali tidak ada blok kode, kembalikan respons aslinya (berisiko error syntax)
        return model_response.strip()

    def run_test(self, source_code, model_response):
        """Menjalankan source code dan unit test yang di-generate model dalam satu file."""
        test_code = self._extract_code(model_response)
        
        # Gabungkan kode asli dan kode testing menjadi satu file agar saling kenal (imports)
        # Dalam skenario dunia nyata, kita harus meletakkan kode asli di module terpisah, 
        # namun untuk kesederhanaan evaluasi, kita tempel di bagian atas.
        combined_code = f"{source_code}\n\n# --- GENERATED TEST BELOW ---\n{test_code}"
        
        # Simpan ke file temporer
        with open(self.temp_filename, "w", encoding="utf-8") as f:
            f.write(combined_code)
            
        try:
            # Jalankan pytest secara diam-diam
            result = subprocess.run(
                ["pytest", self.temp_filename, "-q", "--disable-warnings"],
                capture_output=True,
                text=True,
                timeout=10 # Jangan biarkan test berjalan selamanya jika model membuat infinite loop
            )
            
            # pytest mengembalikan exit code 0 jika semua test lulus
            if result.returncode == 0:
                return True, "Passed"
            else:
                return False, f"{result.stdout}\n\n--- KODE YANG BERUSAHA DIJALANKAN ---\n{combined_code}"
                
        except subprocess.TimeoutExpired:
            return False, f"Timeout: Unit test memakan waktu terlalu lama.\n\n--- KODE ---\n{combined_code}"
        except Exception as e:
            return False, f"System Error: {str(e)}\n\n--- KODE ---\n{combined_code}"
        finally:
            # Bersihkan file temporer
            if os.path.exists(self.temp_filename):
                os.remove(self.temp_filename)
