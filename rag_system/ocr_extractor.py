import fitz
import easyocr
import numpy as np
import os
import time
import sys

# Force UTF-8 untuk menghindari error \u2588 saat download progress bar easyocr di Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def extract_pdf_with_ocr(pdf_path, output_txt_path):
    print(f"Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    print("Loading EasyOCR Model (Indonesian)...")
    reader = easyocr.Reader(['id'], gpu=True)
    
    full_text = ""
    start_time = time.time()
    
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        
        # Render to pixmap with zoom for better OCR accuracy
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # Run OCR
        result = reader.readtext(img, detail=0)
        page_text = "\n".join(result)
        full_text += f"\n\n--- Halaman {page_num+1} ---\n\n" + page_text
        
        elapsed = time.time() - start_time
        avg_time = elapsed / (page_num + 1)
        est_left = avg_time * (total_pages - page_num - 1)
        print(f"✅ OCR Page {page_num + 1}/{total_pages} | Sisa Waktu Est: {est_left/60:.1f} mnt")
        
    print(f"\nMenyimpan hasil ke {output_txt_path} ...")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    print("OCR extraction selesai!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    pdf_path = os.path.join(base_dir, "data", "legal_corpus", "Peraturan-Menteri-Keuangan-Nomor-168-Tahun-2023.pdf")
    output_path = os.path.join(base_dir, "data", "legal_corpus", "PMK_168_OCR.txt")
    
    extract_pdf_with_ocr(pdf_path, output_path)
