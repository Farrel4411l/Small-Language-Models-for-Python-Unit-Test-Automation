import sqlite3
import time
from pathlib import Path
from inference import ModelInferencer
from auto_tester import AutoTester

def load_test_cases(limit=5):
    """Memuat contoh kasus dari database untuk dievaluasi."""
    print(f"📦 Mengambil {limit} sampel data pengujian dari database SQLite...")
    base_dir = Path(__file__).parent.parent
    db_path = base_dir / "data" / "processed" / "payroll_tests.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Mengambil sampel data secara acak.
    cursor.execute("SELECT id, file_name, source_code FROM payroll_qa WHERE is_synthetic = 2 ORDER BY RANDOM() LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    return rows

def main():
    print("="*60)
    print("🚀 MEMULAI FASE 3: AUTOMATED EVALUATION PIPELINE")
    print("="*60)
    
    # 1. Inisialisasi Model dan Auto-Tester
    inferencer = ModelInferencer()
    tester = AutoTester()
    
    # 2. Ambil data pengujian
    N_TESTS = 50
    test_cases = load_test_cases(limit=N_TESTS)
    
    passed_count = 0
    failed_count = 0
    
    print(f"\n🧠 Memulai ujian untuk SLM Spesialis PPh 21 ({N_TESTS} Pertanyaan)...\n")
    
    # 3. Looping Evaluasi
    for i, (row_id, file_name, source_code) in enumerate(test_cases, 1):
        print(f"--- [Test {i}/{N_TESTS}] Ujian Kode dari file: {file_name} ---")
        
        # Step A: Generate kode dengan SLM
        print("Mempersilakan AI berpikir dan menulis kode unit test...")
        start_time = time.time()
        model_response = inferencer.generate_test(source_code)
        elapsed = time.time() - start_time
        print(f"Selesai menulis dalam {elapsed:.2f} detik.")
        
        # Step B: Ekstrak dan jalankan pytest
        print("Menjalankan Auto-Tester (Pytest)...")
        is_passed, log = tester.run_test(source_code, model_response)
        
        if is_passed:
            print("✅ HASIL: LULUS (Semua test berhasil dilewati tanpa error syntax/logika)")
            passed_count += 1
        else:
            print("❌ HASIL: GAGAL")
            print(f"Pesan Error:\n{log}\n")
            failed_count += 1
            
        print("-"*60)
        
    # 4. Hitung Metrik Pass@1
    print("\n" + "="*60)
    print("📊 LAPORAN EVALUASI AKHIR (METRIK LULUS@1)")
    print("="*60)
    print(f"Total Soal Diujikan : {N_TESTS}")
    print(f"Total Lulus (Pass)  : {passed_count}")
    print(f"Total Gagal (Fail)  : {failed_count}")
    
    pass_at_1_score = (passed_count / N_TESTS) * 100
    print(f"\n🏆 AKURASI MODEL (PASS@1) : {pass_at_1_score:.1f}%")
    
    if pass_at_1_score >= 80:
        print("Kesan: LUAR BIASA! SLM Anda setingkat Engineer Senior!")
    elif pass_at_1_score >= 50:
        print("Kesan: BAGUS! SLM Anda sangat mampu menyaingi GPT-3.5!")
    else:
        print("Kesan: Perlu fine-tuning lebih lama atau dataset yang lebih besar.")
    print("="*60)

if __name__ == "__main__":
    main()
