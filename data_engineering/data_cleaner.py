import json
import ast
import os
from pathlib import Path

# Daftar nama file yang pasti tidak mengandung logika penggajian dan harus dibuang
BLACKLIST_FILES = ['setup.py', '__init__.py', 'app.py', 'manage.py', 'tests.py', 'conftest.py']

def extract_functions_from_code(code_string):
    """
    Menggunakan Abstract Syntax Tree (AST) untuk mengekstrak hanya 
    fungsi (def) dan kelas (class) dari sebuah file Python, 
    membuang kode UI, import, dan script level-module lainnya.
    """
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        # Jika kode mengandung syntax error, kita abaikan saja file ini
        return None

    extracted_snippets = []
    
    for node in tree.body:
        # Kita hanya tertarik pada Fungsi atau Class
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # get_source_segment mengambil string kode asli dari node tersebut
            # (Membutuhkan Python 3.8+)
            segment = ast.get_source_segment(code_string, node)
            if segment:
                extracted_snippets.append(segment)
                
    return "\n\n".join(extracted_snippets)

def main():
    base_dir = Path(__file__).parent.parent
    raw_file = base_dir / "data" / "raw" / "github_raw_payroll_data.json"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / "cleaned_payroll_data.json"
    
    if not raw_file.exists():
        print(f"File raw tidak ditemukan: {raw_file}")
        return

    with open(raw_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    print(f"Total data mentah: {len(raw_data)} file.")
    
    cleaned_data = []
    
    for item in raw_data:
        file_path = item.get("file_path", "")
        file_name = os.path.basename(file_path)
        
        # 1. Heuristic Filtering: Buang file blacklist
        if file_name in BLACKLIST_FILES:
            continue
            
        code_content = item.get("code_content", "")
        
        # 2. Buang file yang terlalu pendek (kemungkinan besar tidak ada logika)
        if len(code_content.splitlines()) < 10:
            continue
            
        # 3. AST Parsing: Ambil fungsi/class nya saja
        extracted_code = extract_functions_from_code(code_content)
        
        # Jika file ternyata tidak memiliki fungsi sama sekali (misal cuma deklarasi variabel)
        if not extracted_code or extracted_code.strip() == "":
            continue
            
        # Simpan data yang sudah bersih
        cleaned_item = {
            "repo": item["repo"],
            "file_name": file_name,
            "original_url": item["html_url"],
            "clean_code": extracted_code
        }
        cleaned_data.append(cleaned_item)

    # 4. Simpan ke processed folder
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
        
    print(f"Data bersih berhasil disimpan: {len(cleaned_data)} file.")
    print(f"Lokasi: {output_file}")
    print(f"Berhasil membuang {len(raw_data) - len(cleaned_data)} file sampah/tidak relevan.")

if __name__ == "__main__":
    main()
