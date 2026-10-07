import json
import os
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure nltk resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    raw_out = os.path.join(base_dir, "evaluation", "raw_generation_results.json")
    
    with open(raw_out, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1
    
    total_rouge1 = 0
    total_rougeL = 0
    total_bleu = 0
    
    for item in data:
        ref = item['expected_answer']
        hyp = item['generated_answer']
        
        # ROUGE
        scores = scorer.score(ref, hyp)
        total_rouge1 += scores['rouge1'].fmeasure
        total_rougeL += scores['rougeL'].fmeasure
        
        # BLEU
        ref_tokens = nltk.word_tokenize(ref.lower())
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
        total_bleu += bleu
        
    n = len(data)
    print("=== METRIK EVALUASI TEKS ===")
    print(f"Total Pertanyaan: {n}")
    print(f"ROUGE-1: {total_rouge1 / n:.4f}")
    print(f"ROUGE-L: {total_rougeL / n:.4f}")
    print(f"BLEU:    {total_bleu / n:.4f}")

if __name__ == "__main__":
    main()
