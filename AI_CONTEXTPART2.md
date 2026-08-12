AI-Payroll Project Context & Notes
🎯 Tujuan Utama Riset
Membangun dan mengevaluasi Small Language Model (SLM) berbasis Qwen 2.5 7B Instruct untuk menjadi agen spesialis pajak Indonesia (khususnya PPh 21, PPh 23, PPN, dll). Fokus utama adalah mengukur kemampuan model dalam menghasilkan Unit Test yang benar dan akurat.

🏗️ Struktur Sistem (Pipeline)
Fase 1 (Data Engineering):
Data awal didapatkan dari scraping GitHub (Odoo, Django, Frappe).
Data dibersihkan dan disimpan ke dalam data/processed/payroll_tests.db (SQLite).
Rencana Baru: Augmentasi data menggunakan Procedural Synthetic Generation untuk memperbanyak data yang 100% pure Python (tanpa library).
Fase 2 (Fine-Tuning):
Menggunakan QLoRA (4-bit quantization) melalui PEFT dan TRL.
Melatih model secara lokal di RTX 4070 8GB VRAM (Windows WSL / Ubuntu).
Fase 3 (Automated Evaluation):
Menggunakan subprocess untuk menjalankan Pytest secara dinamis.
Menguji akurasi model menggunakan metrik Pass@1.
💡 Penemuan Kritis & Limitasi (Bahan Skripsi/Jurnal)
Selama Fase 1 hingga 3, kita menemukan beberapa fakta empiris yang sangat penting:

The Probabilistic Trap: LLM menebak probabilitas kata (token), bukan melakukan komputasi deterministik. Oleh karena itu, LLM sering gagal jika disuruh berhitung matematika pajak di kepalanya. Pendekatan yang benar adalah: LLM diminta menulis kode Python/rumus pajaknya, lalu CPU yang menghitungnya (melalui eksekusi kode).
Real-World Code Evaluation Crisis: Mengevaluasi model pada kode hasil scraping (Non-Self-Contained) hampir selalu menghasilkan ImportError atau NameError. Kode dunia nyata terikat pada ekosistemnya (seperti Django models, Odoo osv, Frappe). Ini membuat mesin evaluasi gagal berjalan, meskipun unit test yang di-generate oleh AI secara sintaksis 100% benar.
Halusinasi Looping: Model dengan parameter 7B sangat rentan terhadap Infinite Loop saat menghasilkan teks (misalnya mengimpor library yang sama ratusan kali). Solusinya wajib menggunakan repetition_penalty > 1.0 saat proses Inference.
Token Truncation: Batas max_new_tokens yang terlalu pendek (misal 512) menyebabkan LLM "tercekik" di tengah kalimat, merusak sintaks kode (contoh: kehilangan backtick penutup), yang berujung pada SyntaxError.
🚀 Rencana Fase 4 (Update Part 2)
Menciptakan skrip pembuat dataset sintetis (100% self-contained Python) untuk mengajarkan logika matematis perpajakan tanpa polusi library web.
Menggabungkan data asli (hasil scraping) dan data sintetis agar model tetap mengenali kode dunia nyata, namun memiliki fondasi berhitung yang kuat dari data sintetis.
Memperpanjang waktu training (Epochs) dan menyesuaikan repetition_penalty.