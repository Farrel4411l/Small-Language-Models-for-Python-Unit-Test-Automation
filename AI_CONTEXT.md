# AI Context & Research Notes

**Peneliti:** Farrel
**Topik Utama:** Rekayasa Perangkat Lunak & Kecerdasan Buatan (AI)
**Fokus Spesifik:** Domain-Specific Fine-Tuning untuk Automated Unit Testing (PPh 21 Payroll Logic)

## Ringkasan Proyek
Penelitian ini bertujuan untuk melakukan fine-tuning pada Small Language Model (SLM) seperti Llama-3 8B agar dapat membaca source code Python terkait penggajian (payroll) dan menghasilkan script test (pytest) yang patuh terhadap aturan hukum pajak PPh 21 Indonesia. Hal ini memecahkan masalah privasi data enterprise yang tidak bisa menggunakan API LLM komersial (seperti ChatGPT) untuk memproses logika bisnis dan source code sensitif mereka.

## Struktur Fase Pengembangan (The 3 Phases)

### 1. Data Engineering Pipeline (Menjawab RQ1 / RO1)
Tujuan: Membangun dataset "Semi-Sintetis".
- **Source Code:** Diambil dari repositori open-source Python terkait penggajian/pajak.
- **Test Scripts:** Di-generate menggunakan LLM komersial (GPT-4/Gemini) yang di-prompt agar patuh aturan PPh 21 (misal PTKP, tarif progresif).
- **Infrastruktur:** Disaring menggunakan Relational Database (Supabase/PostgreSQL) dan diekspor menjadi `.jsonl`.

### 2. Fine-Tuning Pipeline (Menjawab RQ3 / RO3)
Tujuan: Melatih model di lingkungan lokal untuk menjamin privasi.
- **Model:** Llama-3 8B (atau sejenisnya).
- **Metode:** QLoRA (Quantized Low-Rank Adaptation) dengan kuantisasi 4-bit (bitsandbytes) untuk efisiensi memori GPU.
- **Infrastruktur:** WSL Ubuntu, CUDA, GPU Consumer-grade.
- **Library:** Hugging Face `transformers`, `peft`, `trl`.

### 3. Evaluation Pipeline (Menjawab RQ2 / RO2)
Tujuan: Menguji performa model yang sudah di fine-tune (Domain-Knowledge Injection).
- **Proses:** Memberikan source code baru, meminta SLM melakukan *generation*, menyimpan sebagai file `.py`, dan mengeksekusinya via `pytest`.
- **Metrics yang diukur:**
    - *Pass rate* (apakah logika pajaknya benar).
    - *Syntax validity* (apakah scriptnya bisa jalan tanpa syntax error).

## Notes to Self (AI Assistant)
- **Konteks:** Setiap kali diminta untuk membantu *coding* di folder manapun di proyek ini, saya (AI) harus selalu ingat bahwa tujuan utamanya adalah kepatuhan pajak PPh 21 (untuk *prompt generation*) dan manajemen privasi/infrastruktur lokal.
- **Gaya Bahasa:** Jika diminta membantu menulis paper/latar belakang, gunakan alur deduktif dan kosakata akademis standar Scopus seperti (Automated Test Generation, Domain-Specific Knowledge, Compliance-Aware, QLoRA).
- **Pendekatan Coding:** Buat skrip yang se-modular mungkin (satu fungsi satu tugas) agar mudah di-debug jika eksperimen machine learning-nya gagal/terjadi Out of Memory (OOM) pada GPU.
