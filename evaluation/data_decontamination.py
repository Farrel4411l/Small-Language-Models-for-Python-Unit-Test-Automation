import sys
import json
import string
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict

base_dir = Path(__file__).parent.parent


def clean_natural_text(text):
    if not text:
        return ""
    # Remove excess whitespace and punctuation for TruthfulQA
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def clean_code_text(text):
    if not text:
        return ""
    # For code, DO NOT remove punctuation; convert to lowercase and normalize whitespace only.
    return " ".join(text.lower().split())


def extract_ngrams(text, n=10):
    words = text.split()
    ngrams = set()
    for i in range(len(words) - n + 1):
        ngrams.add(" ".join(words[i : i + n]))
    return ngrams


def get_local_ngrams_streaming(n=10, is_code=False):
    data_dir = base_dir / "data" / "processed"
    ngrams = set()
    # Stream line by line to minimize RAM usage
    for file_path in data_dir.glob("*.jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    raw_text = data.get("prompt", "") + " " + data.get("response", "")
                    if is_code:
                        clean = clean_code_text(raw_text)
                    else:
                        clean = clean_natural_text(raw_text)
                    ngrams.update(extract_ngrams(clean, n))
                except:
                    pass
    return ngrams


def check_contamination_humaneval(local_ngrams_code, n=10):
    dataset_name = "openai/openai_humaneval"
    split_name = "test"
    print(
        f"\n[Scanning N-Gram={n} Overlap in {dataset_name} (Target: prompt + canonical_solution)]"
    )
    try:
        hf_dataset = load_dataset(dataset_name, split=split_name)
        overlap_count = 0
        total_samples = len(hf_dataset)

        for sample in tqdm(hf_dataset, desc="Scanning HumanEval"):
            # Check prompt and reference solution
            text = (
                str(sample.get("prompt", ""))
                + " "
                + str(sample.get("canonical_solution", ""))
            )
            clean = clean_code_text(text)
            sample_ngrams = extract_ngrams(clean, n)

            if len(local_ngrams_code.intersection(sample_ngrams)) > 0:
                overlap_count += 1

        leakage_percentage = (overlap_count / total_samples) * 100
        print(
            f"-> Contamination detected in {overlap_count}/{total_samples} samples ({leakage_percentage:.2f}%)"
        )
    except Exception as e:
        print(f"Error checking {dataset_name}: {e}")


def check_contamination_truthfulqa(local_ngrams_natural, n=10):
    dataset_name = "truthfulqa/truthful_qa"
    split_name = "validation"
    print(
        f"\n[Scanning N-Gram={n} Overlap in {dataset_name} (Target: question + best_answer + correct_answers + incorrect_answers)]"
    )
    try:
        try:
            hf_dataset = load_dataset(dataset_name, "generation", split=split_name)
        except:
            hf_dataset = load_dataset("truthful_qa", "generation", split=split_name)

        overlap_count = 0
        total_samples = len(hf_dataset)

        for sample in tqdm(hf_dataset, desc="Scanning TruthfulQA"):
            # Check question and all ground truth answer variants
            text = (
                str(sample.get("question", ""))
                + " "
                + str(sample.get("best_answer", ""))
            )

            # correct_answers and incorrect_answers are lists of strings
            if isinstance(sample.get("correct_answers"), list):
                text += " " + " ".join(sample.get("correct_answers"))
            if isinstance(sample.get("incorrect_answers"), list):
                text += " " + " ".join(sample.get("incorrect_answers"))

            clean = clean_natural_text(text)
            sample_ngrams = extract_ngrams(clean, n)

            if len(local_ngrams_natural.intersection(sample_ngrams)) > 0:
                overlap_count += 1

        leakage_percentage = (overlap_count / total_samples) * 100
        print(
            f"-> Contamination detected in {overlap_count}/{total_samples} samples ({leakage_percentage:.2f}%)"
        )
    except Exception as e:
        print(f"Error checking TruthfulQA: {e}")


def main():
    print("=" * 60)
    print("🔍 DATA DECONTAMINATION & LEAKAGE CHECK (STREAMING + KEY CHECK)")
    print("=" * 60)

    print("Extracting Natural N-Grams (for TruthfulQA)...")
    local_ngrams_natural = get_local_ngrams_streaming(n=10, is_code=False)
    print(f"Found {len(local_ngrams_natural)} unique natural 10-grams.")
    check_contamination_truthfulqa(local_ngrams_natural, n=10)

    # Release natural n-grams memory
    local_ngrams_natural.clear()

    print("\nExtracting Code N-Grams (for HumanEval)...")
    local_ngrams_code = get_local_ngrams_streaming(n=10, is_code=True)
    print(f"Found {len(local_ngrams_code)} unique code 10-grams.")
    check_contamination_humaneval(local_ngrams_code, n=10)


if __name__ == "__main__":
    main()
