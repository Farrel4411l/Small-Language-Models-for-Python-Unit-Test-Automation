import sqlite3
import random
import os

def generate_synthetic_data(num_samples=200):
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'payroll_tests.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Tambahkan kolom penanda data sintetis agar mudah dilacak
    try:
        c.execute('ALTER TABLE payroll_qa ADD COLUMN is_synthetic INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass # Kolom sudah ada
        
    func_names = ['hitung_pajak_karyawan', 'kalkulasi_pph21', 'hitung_pph21_bulanan', 'get_tax_amount', 'calculate_tax', 'potongan_pajak']
    var_names = ['gaji', 'penghasilan', 'pendapatan', 'salary', 'gaji_pokok', 'bruto']
    ptkp_vars = ['ptkp', 'bebas_pajak', 'batas_ptkp', 'deduction', 'pengurang']
    
    generated = 0
    while generated < num_samples:
        func = random.choice(func_names)
        var = random.choice(var_names)
        ptkp = random.choice(ptkp_vars)
        
        type_ = random.choice(['single', 'dual', 'ternary'])
        
        if type_ == 'single':
            rate = random.choice([0.05, 0.1, 0.15])
            ptkp_val = random.choice([4500000, 5000000, 54000000])
            source_code = f'''def {func}({var}):
    {ptkp} = {ptkp_val}
    pkp = {var} - {ptkp}
    if pkp <= 0:
        return 0
    return pkp * {rate}
'''
            test_val_1 = ptkp_val - 1000000
            ans_1 = 0
            test_val_2 = ptkp_val + 10000000
            ans_2 = 10000000 * rate
            
            test_code = f'''import pytest

def test_{func}_dibawah_ptkp():
    assert {func}({test_val_1}) == {ans_1}

def test_{func}_diatas_ptkp():
    assert {func}({test_val_2}) == {ans_2}
'''

        elif type_ == 'dual':
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
        pajak_bawah = {limit} * {rate1}
        pajak_atas = (pkp - {limit}) * {rate2}
        return pajak_bawah + pajak_atas
'''
            test_val_1 = 50000000
            ans_1 = 0
            test_val_2 = 64000000 # pkp = 10jt
            ans_2 = 10000000 * rate1
            test_val_3 = 124000000 # pkp = 70jt
            ans_3 = (limit * rate1) + ((70000000 - limit) * rate2)
            
            test_code = f'''import pytest

def test_{func}_nol():
    assert {func}({test_val_1}) == {ans_1}

def test_{func}_tier1():
    assert {func}({test_val_2}) == {ans_2}

def test_{func}_tier2():
    assert {func}({test_val_3}) == {ans_3}
'''
        else:
            source_code = f'''def {func}({var}, status_kawin):
    if status_kawin == 'TK/0':
        {ptkp} = 54000000
    elif status_kawin == 'K/0':
        {ptkp} = 58500000
    else:
        {ptkp} = 54000000
        
    pkp = {var} - {ptkp}
    if pkp <= 0: return 0
    return pkp * 0.05
'''
            ans_tk0 = (60000000 - 54000000) * 0.05
            ans_k0 = (60000000 - 58500000) * 0.05
            test_code = f'''import pytest

def test_{func}_tk0():
    assert {func}(60000000, 'TK/0') == {ans_tk0}

def test_{func}_k0():
    assert {func}(60000000, 'K/0') == {ans_k0}
'''

        # Cek duplikasi agar model tidak menghafal data yang persis sama
        c.execute("SELECT COUNT(*) FROM payroll_qa WHERE source_code = ?", (source_code,))
        if c.fetchone()[0] == 0:
            c.execute("INSERT INTO payroll_qa (source_code, generated_test, is_synthetic) VALUES (?, ?, 1)", (source_code, test_code))
            generated += 1

    conn.commit()
    conn.close()
    print(f"✅ Berhasil menambahkan {num_samples} pasang kode sintetis murni PPh 21 ke database.")

if __name__ == '__main__':
    print("Memulai pembuatan data sintetis...")
    generate_synthetic_data(250)
