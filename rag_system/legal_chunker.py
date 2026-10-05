import os
import re
import json
import pymupdf4llm

def get_legal_corpus_files(corpus_dir):
    files = [f for f in os.listdir(corpus_dir) if f.lower().endswith('.pdf')]
    return [os.path.join(corpus_dir, f) for f in files]

def extract_pasal_chunks(markdown_text, regulation_name):
    """
    Ekstrak teks ke dalam chunk berdasarkan Pasal.
    Menghindari pemecahan di dalam tabel markdown yang di-generate oleh pymupdf4llm.
    """
    # Regex untuk mendeteksi awal Pasal. Format yang sering muncul:
    # "Pasal 1" atau "### Pasal 1" atau "**Pasal 1**"
    # Kita split menggunakan regex lookahead positif
    pasal_pattern = re.compile(r'(?=\n(?:#+\s+)?(?:\*\*)?Pasal\s+\d+(?:\*\*)?)', re.IGNORECASE)
    
    raw_chunks = pasal_pattern.split(markdown_text)
    
    processed_chunks = []
    
    # Chunk pertama biasanya berisi judul, menimbang, mengingat (Preamble)
    if raw_chunks:
        preamble = raw_chunks[0].strip()
        if len(preamble) > 50:
            processed_chunks.append({
                "content": preamble,
                "metadata": {
                    "regulation": regulation_name,
                    "pasal": "Preamble/Judul"
                }
            })
            
    # Proses chunk pasal-pasalnya
    for chunk in raw_chunks[1:]:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # Cari nomor pasal untuk metadata
        match = re.search(r'Pasal\s+(\d+[A-Z]?)', chunk, re.IGNORECASE)
        pasal_no = match.group(1) if match else "Unknown"
        
        # Bersihkan newline ganda berlebih tapi biarkan tabel utuh
        chunk_clean = re.sub(r'\n{3,}', '\n\n', chunk)
        
        processed_chunks.append({
            "content": chunk_clean,
            "metadata": {
                "regulation": regulation_name,
                "pasal": pasal_no
            }
        })
        
    return processed_chunks

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    corpus_dir = os.path.join(base_dir, 'data', 'legal_corpus')
    output_dir = os.path.join(base_dir, 'data', 'processed_chunks')
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'chunks.jsonl')
    
    pdf_files = get_legal_corpus_files(corpus_dir)
    print(f"Ditemukan {len(pdf_files)} file PDF untuk diproses.")
    
    all_chunks = []
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"\nMengekstrak {filename} ...")
        
        try:
            import fitz
            # pymupdf4llm otomatis convert tabel jadi format markdown
            md_text = pymupdf4llm.to_markdown(pdf_path)
            
            # Deteksi jika pymupdf4llm gagal (hanya membaca watermark)
            if len(md_text) < 5000:
                print(f"  [Warning] pymupdf4llm gagal membaca teks asli. Menggunakan fallback PyMuPDF raw extraction...")
                doc = fitz.open(pdf_path)
                md_text = ""
                for page in doc:
                    md_text += page.get_text("text") + "\n\n"
                doc.close()
            
            # Buang bagian index/header yang tidak penting jika perlu (opsional)
            chunks = extract_pasal_chunks(md_text, filename)
            all_chunks.extend(chunks)
            
            print(f"  -> Berhasil mengekstrak {len(chunks)} chunks berstruktur (Pasal-aware).")
        except Exception as e:
            print(f"  -> Gagal memproses {filename}: {e}")
            
    print(f"\nTotal keseluruhan chunks: {len(all_chunks)}")
    
    # Simpan ke jsonl
    print(f"Menyimpan hasil ke {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    print("Selesai! Corpus legal sudah siap untuk di-indexing (Fase 2).")

if __name__ == "__main__":
    main()
