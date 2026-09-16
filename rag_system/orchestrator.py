import re
from .tax_calculator import TaxCalculator

class TaxOrchestrator:
    def __init__(self):
        self.calculator = TaxCalculator()

    def parse_user_intent(self, user_prompt):
        """
        Simple heuristic parser to extract all salaries and PTKP status from casual text.
        """
        prompt_lower = user_prompt.lower()
        
        # 1. Extract Status PTKP (assume one status for all for simplicity)
        status = "TK/0" # Default
        status_keywords = {
            "tk/0": "TK/0", "tk0": "TK/0", "belum nikah": "TK/0", "single": "TK/0", "lajang": "TK/0",
            "tk/1": "TK/1", "tk1": "TK/1", "belum nikah tanggungan 1": "TK/1",
            "tk/2": "TK/2", "tk2": "TK/2",
            "tk/3": "TK/3", "tk3": "TK/3",
            "k/0": "K/0", "k0": "K/0", "nikah": "K/0", "kawin tanpa anak": "K/0",
            "k/1": "K/1", "k1": "K/1", "nikah anak 1": "K/1", "anak 1": "K/1",
            "k/2": "K/2", "k2": "K/2", "nikah anak 2": "K/2", "anak 2": "K/2",
            "k/3": "K/3", "k3": "K/3", "nikah anak 3": "K/3", "anak 3": "K/3"
        }
        for kw, val in status_keywords.items():
            if kw in prompt_lower:
                status = val
                break

        # 2. Extract ALL Salaries (Gaji)
        salaries_monthly = []
        matches_juta = re.findall(r'([\d\.,]+)\s*juta', prompt_lower)
        for m in matches_juta:
            num_str = m.replace(',', '.')
            try:
                salaries_monthly.append(float(num_str) * 1_000_000)
            except:
                pass
                
        matches_miliar = re.findall(r'([\d\.,]+)\s*miliar', prompt_lower)
        for m in matches_miliar:
            num_str = m.replace(',', '.')
            try:
                salaries_monthly.append(float(num_str) * 1_000_000_000)
            except:
                pass
                
        if not salaries_monthly:
            matches_raw = re.findall(r'(\d{6,})', prompt_lower)
            for m in matches_raw:
                try:
                    salaries_monthly.append(float(m))
                except:
                    pass
                    
        # Assume a default if not found (for robustness)
        if not salaries_monthly:
            salaries_monthly = [5000000] # Default 5 Juta
            
        scenarios = []
        for sal in salaries_monthly:
            scenarios.append({
                "gaji_bulanan": sal,
                "gaji_tahunan": sal * 12,
                "status": status
            })
            
        return scenarios

    def process_chat(self, user_prompt, model_inferencer):
        """
        End-to-end wrapper: Parse -> Calculate -> Prompt Build -> LLM Generate
        """
        # Step 1: Parse Intent
        scenarios = self.parse_user_intent(user_prompt)
        
        # Step 2: Build facts
        fakta_text = ""
        data_tahunan = []
        data_bulanan = []
        data_desember = []
        
        for idx, s in enumerate(scenarios, 1):
            gaji_bulanan = s["gaji_bulanan"]
            gaji_tahunan = s["gaji_tahunan"]
            status = s["status"]
            
            pajak_tahunan = round(self.calculator.calculate_tax(gaji_tahunan, status))
            pajak_bulanan, ter_cat, ter_rate = self.calculator.calculate_monthly_tax_ter(gaji_bulanan, status)
            pajak_bulanan = round(pajak_bulanan)
            pajak_desember = round(self.calculator.calculate_december_tax(pajak_tahunan, pajak_bulanan))
            
            fakta_text += f"Skenario {idx}:\n"
            fakta_text += f"- Gaji Tahunan: {round(gaji_tahunan)} | Status: {status} | Pajak Tahunan: {pajak_tahunan}\n"
            fakta_text += f"- Gaji Bulanan: {round(gaji_bulanan)} | Status: {status} | Pajak Bulanan TER: {pajak_bulanan}\n"
            fakta_text += f"- Pajak Desember: {pajak_desember}\n\n"
            
            data_tahunan.append(f"({round(gaji_tahunan)}, '{status}', {pajak_tahunan})")
            data_bulanan.append(f"({round(gaji_bulanan)}, '{status}', {pajak_bulanan})")
            data_desember.append(f"({round(gaji_tahunan)}, '{status}', {pajak_desember})")
            
        str_tahunan = ",\n        ".join(data_tahunan)
        str_bulanan = ",\n        ".join(data_bulanan)
        str_desember = ",\n        ".join(data_desember)
        
        # Step 3: Build System Prompt with RAG injected
        system_prompt = f"""Kamu adalah Asisten AI Perpajakan Profesional. DILARANG merespons dengan narasi panjang, DILARANG membaca file CSV. Hanya berikan SATU blok kode Python berisi 3 unit test menggunakan `@pytest.mark.parametrize`.

[FAKTA MATEMATIKA - DARI KALKULATOR INTERNAL]
{fakta_text}

[INSTRUKSI KODE]
Buatlah kode Python persis dengan struktur berikut, masukkan tuple fakta matematika di atas secara akurat tanpa mengubah atau menyingkat angka nol-nya.

```python
import pytest

@pytest.mark.parametrize('bruto_tahunan, status, expected', [
        {str_tahunan}
])
def test_hitung_pajak_tahunan(bruto_tahunan, status, expected):
    assert hitung_pajak_tahunan(bruto_tahunan, status) == expected

@pytest.mark.parametrize('bruto_bulanan, status, expected', [
        {str_bulanan}
])
def test_hitung_pajak_bulanan_ter(bruto_bulanan, status, expected):
    assert hitung_pajak_bulanan(bruto_bulanan, status) == expected

@pytest.mark.parametrize('bruto_tahunan, status, expected', [
        {str_desember}
])
def test_hitung_pajak_desember(bruto_tahunan, status, expected):
    assert hitung_pajak_desember(bruto_tahunan, status) == expected
```

Tugas Anda hanyalah mereplikasi format kode di atas dengan persis, menyalin angka-angkanya 100% sama (DILARANG berhalusinasi atau mengubah angka sedikitpun).
"""
        # Step 4: Run Inference
        return model_inferencer.generate_general(system_prompt)
