# Fine-Tuning Small Language Models (SLMs) for Tax Domain Unit Testing

![Learning Curve](learning_curve_test.png)

## Overview
This repository contains the official implementation, dataset, and evaluation suite for the research on fine-tuning a 7B parameter Small Language Model (SLM) for Indonesian Tax (PPh 21) computational logic and Pytest generation.

Our research investigates the boundary between **Expertise Transfer** and **Syntax Memorization** in SLMs. By applying Low-Rank Adaptation (LoRA) with 4-bit quantization (QLoRA) on Qwen2.5-7B, we explored the phenomenon of *Grokking*, mitigation of *Catastrophic Forgetting*, and the fundamental limitations of using SLMs as absolute knowledge bases.

## Key Contributions
1. **Domain-Specific Fine-Tuning**: Successfully injected Indonesian PPh21 tax calculation logic into a 7B SLM.
2. **Data Blending (Guardrails)**: Mitigated Catastrophic Forgetting and Sycophancy (Security vulnerabilities) by blending 50% Tax Data, 25% Chit-Chat, and 25% Adversarial Rejections.
3. **Decontamination & Grokking Analysis**: Achieved 0.00% N-Gram overlap (Data Decontamination) and captured the *Grokking* phenomenon across 12,000 steps.
4. **The "Blind Test" Finding**: Proved empirically that without explicit algorithmic prompting (RAG), SLMs suffer from regulatory hallucination, establishing the necessity of Retrieval-Augmented Generation (RAG) for deterministic legal/tax logic.

## Repository Structure
```
.
├── app.py                      # Streamlit interactive frontend and portfolio
├── data/                       # Datasets (Raw, Processed, and Synthetic Generators)
├── data_engineering/           # Data cleaning, database management, and ETL pipelines
├── evaluation/                 # Benchmarking, Blind Tests, and System Metrics scripts
│   ├── inference.py            # Centralized Model Inferencer (LoRA injection)
│   ├── run_real_blind_test.py  # Zero-Shot Out-of-Distribution (OOD) testing
│   └── data_decontamination.py # N-Gram leakage detection
└── fine_tuning/                # Training scripts (QLoRA) and adapter weights
```

## Hardware Profiling (Production Readiness)
- **Peak VRAM Usage**: 5.24 GB (4-bit nf4 Quantization)
- **Inference Throughput**: ~10.74 tokens/second
- **Feasibility**: Fully capable of on-premise edge deployment (e.g., standard laptops) without requiring enterprise A100 GPUs.

## Installation & Setup

1. **Clone the repository and prepare the environment:**
   ```bash
   conda create -n ai-payroll python=3.10
   conda activate ai-payroll
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install streamlit transformers peft bitsandbytes torch
   ```

## Usage

### 1. Interactive Demo (Streamlit)
To launch the UI showcasing the chat interface and research portfolio:
```bash
streamlit run app.py
```

### 2. Running Inference & Blind Tests
To reproduce the real-world OOD (Out-of-Distribution) benchmark that revealed the model's regulatory hallucination:
```bash
python evaluation/run_real_blind_test.py
```

### 3. Evaluating System Metrics & Benchmarks
```bash
python evaluation/run_system_metrics.py
python evaluation/run_public_benchmarks.py
```

## Academic Conclusion
This research concludes that fine-tuning an SLM strictly for legal/tax knowledge bases is architecturally flawed due to **Template Memorization**. The model perfectly masters syntax translation (Text-to-Pytest) but hallucinates regulatory limits (e.g., PTKP boundaries) when challenged with OOD prompts. Therefore, **Retrieval-Augmented Generation (RAG)** is strictly required for legal accuracy, utilizing the fine-tuned SLM purely as a deterministic reasoning and syntax translation engine.
