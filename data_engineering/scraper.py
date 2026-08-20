import os
import time
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (.env file)
load_dotenv()

# Setting GITHUB_TOKEN in the .env file is highly recommended
# to avoid strict GitHub API rate limits (10 requests/minute unauthenticated)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
}

if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

# List of search queries targeting Python code for PPh 21 and Payroll
SEARCH_QUERIES = [
    "PPh 21 language:python",
    "pajak penghasilan 21 language:python",
    "ptkp language:python",
    "penggajian language:python",
    "payroll indonesia language:python",
]


def search_github_code(query, max_results=20):
    """
    Searches GitHub for code snippets matching the given query.
    """
    url = "https://api.github.com/search/code"
    results = []
    page = 1

    while len(results) < max_results:
        params = {
            "q": query,
            "per_page": 30,  # Maximum per page (GitHub Code Search API pagination limit)
            "page": page,
        }

        print(f"Mencari di GitHub (Page {page}) untuk query: '{query}'...")
        response = requests.get(url, headers=HEADERS, params=params)

        if response.status_code == 403:
            print("⚠️ Rate limit API GitHub tercapai, atau token salah.")
            print("Response:", response.json())
            break
        elif response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            break

        data = response.json()
        items = data.get("items", [])

        if not items:
            print("Tidak ada hasil lagi yang ditemukan.")
            break

        for item in items:
            repo_name = item["repository"]["full_name"]
            file_path = item["path"]
            # Convert HTML URL to raw content URL to download original source code
            raw_url = (
                item["html_url"]
                .replace("github.com", "raw.githubusercontent.com")
                .replace("/blob/", "/")
            )

            results.append(
                {
                    "repo": repo_name,
                    "file_path": file_path,
                    "raw_url": raw_url,
                    "html_url": item["html_url"],
                }
            )

            if len(results) >= max_results:
                break

        # Delay 2-3 seconds to comply with GitHub rate limits
        time.sleep(3)
        page += 1

    return results


def download_raw_code(raw_url):
    """
    Downloads raw Python file content from the specified URL.
    """
    try:
        response = requests.get(raw_url)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        print(f"Error saat mengunduh {raw_url}: {e}")
        return None


def main():
    if not GITHUB_TOKEN:
        print("⚠️ PERINGATAN: GITHUB_TOKEN tidak ditemukan di file .env!")
        print(
            "API GitHub akan membatasi pencarian Anda hanya sekitar 10 kali per menit."
        )
        print("Sebaiknya buat Personal Access Token (Classic) di GitHub Anda.\n")

    # Ensure data/raw directory exists
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_snippets = []

    for query in SEARCH_QUERIES:
        print(f"\n--- Memproses Query: {query} ---")
        # Limit results to 10 files per keyword for initial testing
        items = search_github_code(query, max_results=10)

        for item in items:
            print(f"Mengunduh kode: {item['repo']} -> {item['file_path']}")
            code_content = download_raw_code(item["raw_url"])

            if code_content:
                item["code_content"] = code_content
                all_snippets.append(item)

            # Politeness delay for API rate limits
            time.sleep(1)

    # Save all search results into a single JSON file in data/raw
    output_file = raw_dir / "github_raw_payroll_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_snippets, f, indent=4, ensure_ascii=False)

    print(
        f"\n✅ Scraping selesai! Berhasil mengunduh {len(all_snippets)} snippet kode."
    )
    print(f"✅ Data mentah disimpan di: {output_file}")


if __name__ == "__main__":
    main()
