import urllib.request
import os

pdf_links = {
    "PP_58_2023.pdf": "https://jdih.kemenkeu.go.id/download/7487f97a-94ad-4c8d-b0a7-63a5642a8b3d/2023pp058.pdf",
    "PMK_168_2023.pdf": "https://jdih.kemenkeu.go.id/download/e60a82e0-b218-40f5-9d18-b924aa1e11ce/2023pmkeuangan168.pdf",
    "UU_7_2021_HPP.pdf": "https://jdih.kemenkeu.go.id/download/74c7676c-38d5-455a-935f-14838392f446/UU007Tahun2021.pdf",
    "UU_36_2008_PPh.pdf": "https://jdih.kemenkeu.go.id/download/12c19a86-7788-4903-8d2b-d368e7343e06/2008uu036.pdf",
    "PMK_101_2016_PTKP.pdf": "https://jdih.kemenkeu.go.id/download/3074d0e0-c116-430c-ab22-0d50711bf88d/2016pmk010101.pdf",
    "PMK_105_2025_DTP.pdf": "https://jdih.kemenkeu.go.id/download/89930f35-d227-46e3-855f-832145b41054/2025pmkeuangan105.pdf" 
}

output_dir = os.path.dirname(__file__)

for filename, url in pdf_links.items():
    filepath = os.path.join(output_dir, filename)
    print(f"Downloading {filename}...")
    try:
        # Menambahkan User-Agent agar tidak diblokir
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"✅ Success: {filename}")
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
