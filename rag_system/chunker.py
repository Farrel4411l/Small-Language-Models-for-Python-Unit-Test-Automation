import os
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import FlagModel

def test_extraction_and_embedding():
    # 1. Tentukan path PDF
    corpus_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'legal_corpus')
    sample_pdf = os.path.join(corpus_dir, 'PP Nomor 58 Tahun 2023.pdf')
    
    if not os.path.exists(sample_pdf):
        print(f"File tidak ditemukan: {sample_pdf}")
        return

    print(f"--- 1. Mengekstrak PDF dengan PyMuPDF4LLM ---")
    print(f"Membaca {sample_pdf}...")
    # pymupdf4llm mengubah PDF (struktur, tabel, dll) menjadi format Markdown
    md_text = pymupdf4llm.to_markdown(sample_pdf)
    print(f"Berhasil mengekstrak {len(md_text)} karakter (Markdown).")
    print("Pratinjau awal dokumen:\n", md_text[:250], "...\n")

    # 2. Basic Chunking (Sebagai test awal)
    print(f"--- 2. Membagi Teks (Chunking) dengan Langchain ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\nPasal", "\nAyat", "\n\n", "\n", " "]
    )
    chunks = text_splitter.split_text(md_text)
    print(f"Terbagi menjadi {len(chunks)} chunks.")
    print("Pratinjau Chunk 1:\n", chunks[1] if len(chunks) > 1 else chunks[0], "\n")

    # 3. Test Embedding menggunakan BGE-M3
    print(f"--- 3. Test Embedding (BGE-M3) ---")
    print("Memuat model BAAI/bge-m3 (ini mungkin butuh waktu & download pertama kali)...")
    model = FlagModel('BAAI/bge-m3', 
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                      use_fp16=True)
    
    test_chunk = chunks[1] if len(chunks) > 1 else chunks[0]
    embedding = model.encode(test_chunk)
    
    print(f"Dimensi Embedding: {embedding.shape}")
    print(f"5 nilai pertama dari vektor: {embedding[:5]}")
    print("✅ RAG Stack & Test Embedding Berhasil!")

if __name__ == "__main__":
    test_extraction_and_embedding()
