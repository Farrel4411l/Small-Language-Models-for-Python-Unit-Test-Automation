import sqlite3
import random
from pathlib import Path

def setup_database():
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / "data" / "processed" / "payroll_tests.db"
    conn = sqlite3.connect(db_path)
    return conn

def generate_ood_cases():
    """Menghasilkan kasus Out-Of-Distribution (OOD) untuk menguji over-fitting."""
    # Skenario 1: Gaji Negatif atau Nol
    zero_negative = [0, -1000000, -50000, -999999999]
    # Skenario 2: Gaji sangat kecil (di bawah 1 juta)
    micro = [1, 50, 1000, 500000, 999999]
    # Skenario 3: Gaji ekstrem besar (Ratusan Miliar hingga Triliunan)
    massive = [5000000000, 10000000000, 999999999999, 5000000000000]
    # Skenario 4: Pecahan desimal yang tidak wajar di industri
    decimal = [10500000.12345, 60000000.999, 1500000.0001, -500.5]
    
    all_salaries = zero_negative + micro + massive + decimal
    # Tambahkan random OOD untuk melengkapi jadi ~100 data
    for _ in range(30):
        all_salaries.append(random.uniform(-1000000, 0))
    for _ in range(30):
        all_salaries.append(random.uniform(5000000000, 10000000000))
    for _ in range(20):
        all_salaries.append(random.uniform(1000, 999999))
        
    return all_salaries

def create_prompt_and_test(salary):
    # Format persis seperti data training (Formula-based)
    source_code = f"""def potongan_bulanan(bruto):
    pengurang = 63000000
    pkp = bruto - pengurang
    if pkp <= 0:
        return 0
    return pkp * 0.15"""

    # Karena ini OOD, kita harus menyediakan jawaban (test) yang benar dengan format formula
    if salary <= 63000000:
        expected = "0"
    else:
        expected = f"({salary} - 63000000) * 0.15"

    test_code = f"""import pytest

def test_potongan_bulanan_ood():
    assert potongan_bulanan({salary}) == {expected}
"""
    return source_code, test_code

def main():
    print("Mempersiapkan dataset OOD (Out-Of-Distribution)...")
    conn = setup_database()
    cursor = conn.cursor()
    
    # Hapus OOD test sebelumnya jika ada
    cursor.execute("DELETE FROM payroll_qa WHERE split_type = 'ood_test'")
    
    salaries = generate_ood_cases()
    print(f"Mengenerate {len(salaries)} kasus ekstrem...")
    
    for idx, salary in enumerate(salaries):
        source_code, generated_test = create_prompt_and_test(salary)
        
        cursor.execute('''
            INSERT INTO payroll_qa (source_code, generated_test, is_synthetic, split_type)
            VALUES (?, ?, 1, ?)
        ''', (source_code, generated_test, 'ood_test'))
        
    conn.commit()
    conn.close()
    print("Selesai! Data 'ood_test' telah dimasukkan ke database.")

if __name__ == "__main__":
    main()
