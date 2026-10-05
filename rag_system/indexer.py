import os
import json
import pickle
import chromadb
from FlagEmbedding import FlagModel
from rank_bm25 import BM25Okapi

def load_chunks(jsonl_path):
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def get_bm25_tokens(text):
    # Tokenisasi sederhana untuk BM25
    return text.lower().split()

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    chunks_file = os.path.join(base_dir, 'data', 'processed_chunks', 'chunks.jsonl')
    db_dir = os.path.join(base_dir, 'data', 'vector_db')
    bm25_file = os.path.join(base_dir, 'data', 'vector_db', 'bm25_index.pkl')
    
    os.makedirs(db_dir, exist_ok=True)
    
    print("1. Memuat processed chunks...")
    chunks = load_chunks(chunks_file)
    print(f"Total chunks dimuat: {len(chunks)}")
    
    print("\n2. Menginisialisasi Model Embedding BAAI/bge-m3...")
    # BGE-M3 model for Dense Embedding
    model = FlagModel('BAAI/bge-m3', use_fp16=True)
    
    print("\n3. Menyiapkan ChromaDB (Vector Store Lokal)...")
    chroma_client = chromadb.PersistentClient(path=db_dir)
    # Recreate collection to ensure it's fresh
    try:
        chroma_client.delete_collection(name="legal_corpus_bge_m3")
    except Exception:
        pass
    collection = chroma_client.create_collection(name="legal_corpus_bge_m3")
    
    print("\n4. Membangun Sparse Index (BM25)...")
    corpus_tokens = [get_bm25_tokens(chunk['content']) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    
    # Simpan model BM25
    with open(bm25_file, 'wb') as f:
        pickle.dump({
            "bm25_model": bm25,
            "chunks_data": chunks
        }, f)
    print("BM25 Index disimpan ke:", bm25_file)
    
    print("\n5. Mengekstraksi Dense Embeddings dan menyimpannya ke ChromaDB...")
    documents = []
    metadatas = []
    ids = []
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        content = chunk['content']
        metadata = chunk['metadata']
        
        # Inject chunk ID ke metadata agar sinkron dengan BM25
        chunk_id = f"chunk_{i}"
        metadata['chunk_id'] = chunk_id
        
        documents.append(content)
        metadatas.append(metadata)
        ids.append(chunk_id)
        
        # Cetak progress
        if (i + 1) % 50 == 0 or (i + 1) == len(chunks):
            print(f"  Proses {i + 1}/{len(chunks)} chunks...")
            
    # Generate embeddings sekaligus (batch processing)
    # BGE-M3 encode function handles batches efficiently
    print("  Sedang melakukan komputasi embeddings (GPU/CPU)... ini mungkin butuh beberapa menit.")
    vector_embeddings = model.encode(documents)
    embeddings_list = vector_embeddings.tolist()
    
    print("  Menyimpan embeddings ke ChromaDB...")
    # Add to ChromaDB in batches to avoid size limits
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            embeddings=embeddings_list[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        
    print(f"\n✅ FASE 2 SELESAI: {len(chunks)} chunks berhasil di-index!")
    print(f"Total Dense Vectors di ChromaDB: {collection.count()}")

if __name__ == "__main__":
    main()
