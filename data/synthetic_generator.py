import sqlite3
import random
import os


def reset_and_generate_data(num_samples=2000):
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "processed",
        "payroll_tests.db",
    )
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Recreate table schema with split_type column
    c.execute("DROP TABLE IF EXISTS payroll_qa")
    c.execute("""
        CREATE TABLE payroll_qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            source_code TEXT,
            generated_test TEXT,
            is_synthetic INTEGER DEFAULT 1,
            split_type TEXT
        )
    """)

    func_names = [
        "hitung_pajak_karyawan",
        "kalkulasi_pph21",
        "hitung_pph21_bulanan",
        "get_tax_amount",
        "calculate_tax",
        "potongan_pajak",
        "hitung_pph_final",
        "cek_pajak",
        "potongan_bulanan",
        "tax_calc",
        "hitung_pajak",
        "pajak_tahunan",
        "pajak_netto",
    ]
    var_names = [
        "gaji",
        "penghasilan",
        "pendapatan",
        "salary",
        "gaji_pokok",
        "bruto",
        "take_home_pay",
        "thp",
        "gaji_bersih",
    ]
    ptkp_vars = [
        "ptkp",
        "bebas_pajak",
        "batas_ptkp",
        "deduction",
        "pengurang",
        "potongan_tidak_kena_pajak",
        "nilai_ptkp",
    ]

    generated = 0
    while generated < num_samples:
        func = random.choice(func_names)
        var = random.choice(var_names)
        ptkp = random.choice(ptkp_vars)

        type_ = random.choice(["single", "dual"])

        # Dynamically assign dataset split
        rand_val = random.random()
        if rand_val < 0.8:
            split = "train"
        elif rand_val < 0.9:
            split = "val"
        else:
            split = "test"

        if type_ == "single":
            rate = random.choice([0.05, 0.1, 0.15])
            ptkp_val = random.choice([54000000, 58500000, 63000000, 67500000])
            source_code = f"""def {func}({var}):
    {ptkp} = {ptkp_val}
    pkp = {var} - {ptkp}
    if pkp <= 0:
        return 0
    return pkp * {rate}
"""
            test_val_1 = ptkp_val - random.randint(1000000, 5000000)
            test_val_2 = ptkp_val + random.randint(10000000, 50000000)

            # --- FORMULA-BASED ASSERT ---
            test_code = f"""import pytest

def test_{func}_dibawah_ptkp():
    assert {func}({test_val_1}) == 0

def test_{func}_diatas_ptkp():
    assert {func}({test_val_2}) == ({test_val_2} - {ptkp_val}) * {rate}
"""

        else:
            rate1, rate2 = 0.05, 0.15
            limit = 60000000
            ptkp_val = 54000000
            source_code = f"""def {func}({var}):
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
"""
            test_val_1 = 50000000
            test_val_2 = 64000000  # Taxable income (PKP) = 10M
            test_val_3 = 124000000  # Taxable income (PKP) = 70M

            # --- FORMULA-BASED ASSERT ---
            test_code = f"""import pytest

def test_{func}_nol():
    assert {func}({test_val_1}) == 0

def test_{func}_tier1():
    assert {func}({test_val_2}) == ({test_val_2} - {ptkp_val}) * {rate1}

def test_{func}_tier2():
    tier1_tax = {limit} * {rate1}
    tier2_tax = ({test_val_3} - {ptkp_val} - {limit}) * {rate2}
    assert {func}({test_val_3}) == tier1_tax + tier2_tax
"""

        # Check for duplicates to prevent the model from memorizing identical samples
        c.execute(
            "SELECT COUNT(*) FROM payroll_qa WHERE source_code = ?", (source_code,)
        )
        if c.fetchone()[0] == 0:
            c.execute(
                "INSERT INTO payroll_qa (source_code, generated_test, is_synthetic, split_type) VALUES (?, ?, 1, ?)",
                (source_code, test_code, split),
            )
            generated += 1

    conn.commit()

    # Compute split distribution statistics
    c.execute("SELECT split_type, COUNT(*) FROM payroll_qa GROUP BY split_type")
    stats = c.fetchall()
    conn.close()

    print("✅ Successfully rebuilt the database!")
    print(f"Total Samples: {num_samples}")
    for split_type, count in stats:
        print(f" - {split_type.upper()}: {count} rows")


if __name__ == "__main__":
    print("Starting large-scale synthetic data generation (Formula-Based)...")
    reset_and_generate_data(2000)
