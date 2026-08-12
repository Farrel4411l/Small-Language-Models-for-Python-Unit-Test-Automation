import sqlite3
import random
import os

def generate_holdout_data(num_samples=50):
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'payroll_tests.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    func_names = ['hitung_pajak_baru', 'kalkulasi_pph_final', 'cek_potongan_pajak', 'tax_calculator_unseen', 'pajak_karyawan_tetap']
    var_names = ['total_gaji', 'pendapatan_bruto', 'take_home_pay', 'thp', 'gaji_kotor']
    ptkp_vars = ['nilai_ptkp', 'potongan_tidak_kena_pajak', 'bebasan', 'pengurang_pajak']
    
    generated = 0
    while generated < num_samples:
        func = random.choice(func_names)
        var = random.choice(var_names)
        ptkp = random.choice(ptkp_vars)
        
        type_ = random.choice(['single', 'dual'])
        
        if type_ == 'single':
            rate = random.choice([0.05, 0.1, 0.15])
            ptkp_val = random.choice([54000000, 58500000, 63000000, 67500000]) # TK/0, K/0, K/1, K/2
            source_code = f'''def {func}({var}):
    {ptkp} = {ptkp_val}
    pkp = {var} - {ptkp}
    if pkp <= 0:
        return 0
    return pkp * {rate}
'''
            test_val_1 = ptkp_val - 2000000
            ans_1 = 0
            test_val_2 = ptkp_val + 20000000
            ans_2 = 20000000 * rate
            
            test_code = f'''import pytest

def test_{func}_nihil():
    assert {func}({test_val_1}) == {ans_1}

def test_{func}_kena_pajak():
    assert {func}({test_val_2}) == {ans_2}
'''
        else:
            rate1, rate2 = 0.05, 0.15
            limit = 60000000
            ptkp_val = 54000000
            source_code = f'''def {func}({var}):
    {ptkp} = {ptkp_val}
    pkp = {var} - {ptkp}
    if pkp <= 0:
        return 0
    if pkp <= {limit}:
        return pkp * {rate1}
    else:
        tier1 = {limit} * {rate1}
        tier2 = (pkp - {limit}) * {rate2}
        return tier1 + tier2
'''
            test_val_1 = 50000000
            ans_1 = 0
            test_val_2 = 70000000 # pkp = 16jt
            ans_2 = 16000000 * rate1
            test_val_3 = 130000000 # pkp = 76jt
            ans_3 = (limit * rate1) + ((76000000 - limit) * rate2)
            
            test_code = f'''import pytest

def test_{func}_bebas():
    assert {func}({test_val_1}) == {ans_1}

def test_{func}_tier_bawah():
    assert {func}({test_val_2}) == {ans_2}

def test_{func}_tier_atas():
    assert {func}({test_val_3}) == {ans_3}
'''

        # Cek duplikasi agar model tidak menghafal data yang persis sama
        c.execute("SELECT COUNT(*) FROM payroll_qa WHERE source_code = ?", (source_code,))
        if c.fetchone()[0] == 0:
            # is_synthetic = 2 menandakan data khusus HOLDOUT TEST
            c.execute("INSERT INTO payroll_qa (source_code, generated_test, is_synthetic) VALUES (?, ?, 2)", (source_code, test_code))
            generated += 1

    conn.commit()
    conn.close()
    print(f"✅ Berhasil menambahkan {num_samples} data Hold-out Test (is_synthetic=2) ke database.")

if __name__ == '__main__':
    generate_holdout_data(50)
