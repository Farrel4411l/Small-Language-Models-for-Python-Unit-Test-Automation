import sqlite3
import random
from pathlib import Path

def setup_database():
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / "data" / "processed" / "payroll_tests.db"
    conn = sqlite3.connect(db_path)
    return conn

def generate_extreme_salaries():
    """Menghasilkan gaji ekstrem."""
    zero_negative = [0, -1000000, -50000, -999999999]
    micro = [1, 50, 1000, 500000, 999999]
    massive = [5000000000, 10000000000, 999999999999, 5000000000000]
    decimal = [10500000.12345, 60000000.999, 1500000.0001, -500.5]
    
    all_salaries = zero_negative + micro + massive + decimal
    for _ in range(15):
        all_salaries.append(random.uniform(-1000000, 0))
    for _ in range(15):
        all_salaries.append(random.uniform(5000000000, 10000000000))
        
    return all_salaries

def create_numeric_prompt_and_test(salary):
    source_code = f"""def potongan_bulanan(bruto):
    pengurang = 63000000
    pkp = bruto - pengurang
    if pkp <= 0:
        return 0
    return pkp * 0.15"""

    if salary <= 63000000:
        expected = "0"
    else:
        expected = f"({salary} - 63000000) * 0.15"

    test_code = f"""import pytest\n\ndef test_potongan_bulanan_ood():\n    assert potongan_bulanan({salary}) == {expected}\n"""
    return source_code, test_code

def create_linguistic_prompt_and_test(salaries):
    cases = []
    
    for salary in salaries:
        expected = "0" if salary <= 54000000 else f"({salary} - 54000000) * 0.1"
        
        # 1. Typo / Slang
        cases.append((
            f"""# bang tolongin buatin fngsi pph dong\n# ptkp nya 54jt yak klo dbwah nol gausah bayar, potongan 10prsn\ndef ngitung_pajak(gaji_kotor):
    ptkp = 54000000
    sisa = gaji_kotor - ptkp
    if sisa <= 0: return 0
    return sisa * 0.1""",
            f"""import pytest\n\ndef test_ngitung_pajak_slang():\n    assert ngitung_pajak({salary}) == {expected}\n"""
        ))
        
        # 2. Narrative (Word Problem)
        cases.append((
            f"""# Pak Budi adalah seorang karyawan dengan PTKP senilai 54,000,000.
# Setiap bulannya ia dikenakan tarif pajak sebesar 10% dari Penghasilan Kena Pajaknya.
# Jika penghasilannya tidak melebihi PTKP, ia dibebaskan dari pajak.
def kalkulasi_pajak_pak_budi(penghasilan):
    batas_ptkp = 54000000
    pkp = penghasilan - batas_ptkp
    if pkp <= 0:
        return 0
    return pkp * 0.1""",
            f"""import pytest\n\ndef test_kalkulasi_pajak_pak_budi_narrative():\n    assert kalkulasi_pajak_pak_budi({salary}) == {expected}\n"""
        ))
        
        # 3. JSON Format Input / Output instruction masquerading as docstring
        cases.append((
            f"""def calculate_tax_json(income_data):
    \"\"\"
    Input: {{"salary": int/float}}
    Output: {{"tax": float}}
    Rule: PTKP is 54000000, rate is 10%.
    \"\"\"
    ptkp = 54000000
    pkp = income_data["salary"] - ptkp
    if pkp <= 0:
        return {{"tax": 0}}
    return {{"tax": pkp * 0.1}}""",
            f"""import pytest\n\ndef test_calculate_tax_json():\n    assert calculate_tax_json({{"salary": {salary}}}) == {{"tax": {expected}}}\n"""
        ))
        
    return cases

def main():
    print("Mempersiapkan dataset TRIAD OOD...")
    conn = setup_database()
    cursor = conn.cursor()
    
    # Hapus data sebelumnya
    cursor.execute("DELETE FROM payroll_qa WHERE split_type IN ('ood_test', 'ood_math', 'ood_linguistic', 'ood_mixed')")
    
    # 1. Generate OOD Math (Formal language, extreme salaries)
    extreme_salaries = generate_extreme_salaries()
    math_count = 0
    for salary in extreme_salaries:
        source_code, generated_test = create_numeric_prompt_and_test(salary)
        cursor.execute("INSERT INTO payroll_qa (source_code, generated_test, is_synthetic, split_type) VALUES (?, ?, 1, 'ood_math')", (source_code, generated_test))
        math_count += 1
        
    # 2. Generate OOD Linguistic (Slang/Narrative, normal salaries)
    normal_salaries = [50000000, 80000000, 100000000, 150000000, 20000000]
    linguistic_cases = create_linguistic_prompt_and_test(normal_salaries)
    ling_count = 0
    for source_code, generated_test in linguistic_cases:
        cursor.execute("INSERT INTO payroll_qa (source_code, generated_test, is_synthetic, split_type) VALUES (?, ?, 1, 'ood_linguistic')", (source_code, generated_test))
        ling_count += 1
        
    # 3. Generate OOD Mixed (Slang/Narrative, extreme salaries)
    mixed_salaries = [0, -1000000, 1, 999999999999, 10500000.12345]
    mixed_cases = create_linguistic_prompt_and_test(mixed_salaries)
    mixed_count = 0
    for source_code, generated_test in mixed_cases:
        cursor.execute("INSERT INTO payroll_qa (source_code, generated_test, is_synthetic, split_type) VALUES (?, ?, 1, 'ood_mixed')", (source_code, generated_test))
        mixed_count += 1
        
    conn.commit()
    conn.close()
    print(f"Selesai! Dimasukkan:\n- {math_count} soal OOD Math\n- {ling_count} soal OOD Linguistic\n- {mixed_count} soal OOD Mixed")

if __name__ == "__main__":
    main()
