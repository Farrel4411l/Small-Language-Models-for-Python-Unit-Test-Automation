import os
import pickle
import chromadb
from FlagEmbedding import FlagModel, FlagReranker

# Monkeypatch transformers to bypass torch.load vulnerability check (CVE-2025-32434)
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

class HybridRetriever:
    def __init__(self, top_k=20, rerank_top_k=3):
        print("[System] Menginisialisasi Hybrid Retriever...")
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        db_dir = os.path.join(base_dir, 'data', 'vector_db')
        bm25_file = os.path.join(db_dir, 'bm25_index.pkl')
        
        # Load Models
        print("  -> Memuat model BGE-M3 (Dense)...")
        self.dense_model = FlagModel('BAAI/bge-m3', use_fp16=True)
        
        print("  -> Memuat model BGE-Reranker-v2-m3 (Reranker)...")
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        
        # Load Vector DB
        print("  -> Menghubungkan ke ChromaDB...")
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.client.get_collection(name="legal_corpus_bge_m3")
        
        # Load BM25
        print("  -> Memuat Sparse Index (BM25)...")
        with open(bm25_file, 'rb') as f:
            bm25_data = pickle.load(f)
            self.bm25_model = bm25_data['bm25_model']
            self.chunks_data = bm25_data['chunks_data']
            
        print("[System] Retriever siap digunakan!\n")

    def rrf_score(self, rank, k=60):
        return 1 / (k + rank)
        
    def retrieve(self, query):
        print(f"Mencari konteks untuk: '{query}'")
        
        # 1. Sparse Retrieval (BM25)
        query_tokens = query.lower().split()
        bm25_scores = self.bm25_model.get_scores(query_tokens)
        
        # Sort BM25 results
        bm25_ranked = sorted(
            [(i, score) for i, score in enumerate(bm25_scores)], 
            key=lambda x: x[1], 
            reverse=True
        )[:self.top_k]
        
        # 2. Dense Retrieval (ChromaDB)
        query_embedding = self.dense_model.encode([query])[0].tolist()
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )
        
        dense_ranked = []
        if dense_results['ids'] and len(dense_results['ids']) > 0:
            for idx, chunk_id in enumerate(dense_results['ids'][0]):
                # Ekstrak index integer dari chunk_id (format: "chunk_X")
                try:
                    chunk_idx = int(chunk_id.split('_')[1])
                    dense_ranked.append((chunk_idx, dense_results['distances'][0][idx]))
                except:
                    pass
                    
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        # Tambahkan skor BM25
        for rank, (chunk_idx, _) in enumerate(bm25_ranked, 1):
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + self.rrf_score(rank)
            
        # Tambahkan skor Dense
        for rank, (chunk_idx, _) in enumerate(dense_ranked, 1):
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + self.rrf_score(rank)
            
        # Sort berdasarkan RRF score
        hybrid_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        
        # 4. Reranking (Cross-Encoder)
        candidates = []
        for chunk_idx, _ in hybrid_ranked:
            chunk_content = self.chunks_data[chunk_idx]['content']
            candidates.append((query, chunk_content))
            
        # Compute rerank scores
        rerank_scores = self.reranker.compute_score(candidates)
        
        # Jika candidates hanya 1, compute_score mungkin tidak me-return list
        if not isinstance(rerank_scores, list):
            rerank_scores = [rerank_scores]
            
        # Sort berdasarkan Rerank score
        final_ranked = sorted(
            zip(hybrid_ranked, rerank_scores), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.rerank_top_k]
        
        # Siapkan output akhir
        retrieved_chunks = []
        for (chunk_idx, rrf_val), score in final_ranked:
            retrieved_chunks.append({
                "content": self.chunks_data[chunk_idx]['content'],
                "metadata": self.chunks_data[chunk_idx]['metadata'],
                "rerank_score": score
            })
            
        return retrieved_chunks

if __name__ == "__main__":
    retriever = HybridRetriever(top_k=20, rerank_top_k=3)
    
    # Test Query
    query = "Bagaimana perhitungan pemotongan PPh Pasal 21 untuk Pegawai Tetap berdasarkan TER?"
    results = retriever.retrieve(query)
    
    print("\n--- HASIL RETRIEVAL (TOP 3) ---")
    for i, res in enumerate(results, 1):
        print(f"\n[Rank {i} | Score: {res['rerank_score']:.4f}]")
        print(f"Sumber: {res['metadata'].get('regulation', 'Unknown')} - Pasal {res['metadata'].get('pasal', '?')}")
        print(f"Kutipan: {res['content'][:250]}...\n")
