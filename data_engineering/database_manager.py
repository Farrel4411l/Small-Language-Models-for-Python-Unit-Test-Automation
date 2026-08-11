import sqlite3
import json
from pathlib import Path

def setup_database(db_path):
    """
    Membuat koneksi ke database SQLite lokal dan membuat tabel jika belum ada.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Membuat skema tabel
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS payroll_qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo TEXT,
            file_name TEXT,
            source_code TEXT,
            generated_test TEXT,
            is_valid BOOLEAN DEFAULT 0
        )
    ''')
    
    conn.commit()
    return conn, cursor

def main():
    base_dir = Path(__file__).parent.parent
    json_file = base_dir / "data" / "processed" / "qa_dataset.json"
    db_file = base_dir / "data" / "processed" / "payroll_tests.db"
    
    if not json_file.exists():
        print(f"File JSON tidak ditemukan: {json_file}")
        return
        
    # Setup database
    conn, cursor = setup_database(db_file)
    print(f"Database SQLite berhasil dikonfigurasi di: {db_file}")
    
    # Load data dari JSON
    with open(json_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Mengimpor {len(dataset)} data ke dalam database...")
    
    inserted_count = 0
    for item in dataset:
        # Cek duplikasi berdasarkan repo dan file_name
        cursor.execute('SELECT id FROM payroll_qa WHERE repo = ? AND file_name = ?', 
                      (item['repo'], item['file_name']))
        
        if cursor.fetchone() is None:
            cursor.execute('''
                INSERT INTO payroll_qa (repo, file_name, source_code, generated_test)
                VALUES (?, ?, ?, ?)
            ''', (item['repo'], item['file_name'], item['source_code'], item['generated_test']))
            inserted_count += 1
            
    conn.commit()
    conn.close()
    
    print(f"Berhasil menambahkan {inserted_count} data baru ke database SQLite.")
    print("Tahap Data Engineering (Fase 1) SELESAI!")

if __name__ == "__main__":
    main()
