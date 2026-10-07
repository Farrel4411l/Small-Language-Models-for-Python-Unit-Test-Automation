# RAG Implementation Plan: Tax Domain SLM

Dokumen ini adalah *Master Plan* teknis untuk implementasi 100% Local RAG guna menyelesaikan masalah halusinasi regulasi pada Qwen2.5-7B, sesuai dengan target publikasi jurnal Q1 (IEEE Access).

## 🎯 Tujuan Utama
Menguji secara empiris: *"Apakah SLM (Small Language Model) mampu melakukan factual reasoning atas teks regulasi statutory yang di-retrieve?"* (Bukan sekadar menyalin angka dari sistem deterministik).

## 🏗️ Arsitektur Target
* **Embedder:** BAAI/bge-m3 (Dense) + rank_bm25 (Sparse)
* **Vector DB:** ChromaDB (Local Persist)
* **Reranker:** BAAI/bge-reranker-v2-m3
* **Generator:** Qwen2.5-7B-Instruct (Fine-Tuned LoRA)
* **Evaluator:** RAGAS (Faithfulness, Answer Relevance) + Pass@1 (Pytest AST)

---

## 📅 Roadmap Implementasi Rinci

### Fase 1: Corpus Engineering (Current Phase)
Fokus: *Chunking* teks legal yang *structure-aware* (kesalahan chunking = kegagalan RAG).
- [x] **1.1** Ekstraksi PDF menggunakan `pymupdf4llm` untuk mempertahankan format tabel (TER/PTKP).
- [x] **1.2** *Structure-aware chunking*: Membuat pemisah khusus berbasis *Regex* yang mematuhi batas hirarki **BAB -> Pasal -> Ayat**. Tabel tarif tidak boleh terpotong. (195 chunks berhasil diekstrak).
- [x] **1.3** Injeksi Metadata: Setiap *chunk* wajib memiliki metadata `{"regulasi": "PP 58/2023", "pasal": "17", "ayat": "1"}`.
- [ ] **1.3b** **[DECISION LOG]** PMK 168 Tahun 2023 di-skip sementara dari pipeline karena seluruh rilis Kemenkeu/BPK merupakan hasil *scan* gambar tanpa teks digital. PMK 168 akan dimasukkan belakangan setelah kita membangun *pipeline OCR (Optical Character Recognition)* khusus untuk membacanya. RAG akan berjalan dengan 6 Corpus utama untuk saat ini.
- [ ] **1.4** Normalisasi numerik dan teks (misal: "Rp54.000.000" -> "Rp 54.000.000").
- [ ] **1.5** Membangun daftar pertanyaan Emas (*Gold-standard chunks*) secara manual.

### Fase 2: Retrieval Pipeline
Fokus: Menggabungkan Sparse + Dense untuk akurasi tinggi.
- [x] **2.1** Indexing teks ke dalam ChromaDB menggunakan `BGE-M3`. (281 chunks selesai di-index).
- [x] **2.2** Indexing teks menggunakan algoritma `rank_bm25` (disimpan sbg file `.pkl`).
- [ ] **2.3** Membangun *Hybrid Retriever* menggunakan algoritma *Reciprocal Rank Fusion* (RRF) (k=60 -> Top 20).
- [ ] **2.4** Membangun layer *Reranker* dengan `bge-reranker-v2-m3` untuk memeras Top 20 -> Top 3.
- [ ] **2.5** Pengujian Retrieval: Recall@3 ≥ 0.85 pada *gold set*.

### Fase 3: Generation Pipeline
Fokus: Integrasi Prompt RAG dengan Model Qwen2.5-7B.
- [x] **3.1** Desain Template Prompt (*System* + *Context* + *User Query*).
- [x] **3.2** Manajemen Konteks: Menjaga total *token* di bawah batas maksimum model (6K budget utk konteks).
- [x] **3.3** *Citation-Forcing*: Modifikasi prompt agar SLM selalu mengutip Pasal & Regulasi yang ia gunakan sebagai dasar.
- [x] **3.4** Penggabungan dengan Adapter LoRA Qwen yang sudah di-*fine-tune*.

### Fase 4: Evaluation Matrix (Q1 Focus)
Fokus: Pembuktian Ilmiah terhadap 6 Kondisi (Harus dijalankan dengan model asli di GPU).
- [ ] **4.1** Evaluasi C1 (Baseline Qwen).
- [ ] **4.2** Evaluasi C2 (FT-Only, *Current State*).
- [ ] **4.3** Evaluasi C3 (Qwen Raw + RAG).
- [ ] **4.4** Evaluasi C4 (FT + RAG) -> *Novelty Utama*.
- [ ] **4.5** Evaluasi C5 (Tool-only) & C6 (Ultimate).
- [ ] **4.6** Eksekusi evaluasi metrik RAGAS (Faithfulness) secara lokal.

### Fase 5: Ablation Studies & Paper Writing
- [ ] **5.1** *Ablation:* Kinerja Retriever (Dense vs Sparse vs Hybrid) menggunakan Recall@3.
- [ ] **5.2** *Ablation:* Jumlah *k* (1, 3, 5, 10).
- [ ] **5.3** Draft Paper, Pembuatan Grafik Matplotlib beresolusi tinggi, Penyusunan *Methodology* berdasarkan **DATA ASLI**.

---
*Status: Aktif dieksekusi oleh Agent AI Antigravity.*
