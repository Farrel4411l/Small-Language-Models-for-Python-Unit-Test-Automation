import streamlit as st
import time
import os
import torch
from pathlib import Path
from PIL import Image
import sys

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI Payroll Tax SLM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === ABLATION STUDY SELECTION (C1 - C6) ===
st.sidebar.title("⚙️ Arsitektur RAG")
st.sidebar.markdown("Pilih kondisi eksperimen (Ablation Study) yang ingin Anda demonstrasikan:")

c_options = {
    "C1: Baseline (Qwen Raw)": "Model Qwen dasar tanpa modifikasi apa pun. Sangat rawan halusinasi regulasi pajak.",
    "C2: FT-Only (LoRA)": "Model yang sudah di-finetune (Kondisi lama). Pintar menulis *pytest*, tapi menghafal pajak sehingga sering salah hitung.",
    "C3: Baseline + RAG": "Qwen mentah yang diberi asupan teks hukum perpajakan asli (PMK/UU).",
    "C4: FT + RAG (Ours)": "Model pintar coding + asupan teks hukum faktual. Ini adalah **novelty riset kita**, menurunkan halusinasi secara drastis.",
    "C5: Tool-only (Kalkulator)": "Kalkulator deterministik konvensional (Hardcoded Python) tanpa *reasoning* AI.",
    "C6: Ultimate (FT + RAG + Tool)": "Sistem pamungkas! Kombinasi dari bahasa luwes SLM, konteks hukum faktual (RAG), dan kalkulasi akurat (Python Tool)."
}

selected_c = st.sidebar.radio("Pilih Kondisi (C1 - C6):", list(c_options.keys()), index=5)
st.sidebar.success(f"**Info {selected_c.split(':')[0]}:**\n{c_options[selected_c]}")

# Attempt to import ModelInferencer (Only when running locally with GPU support)
error_msg = ""
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from evaluation.inference import ModelInferencer
    from rag_system.orchestrator import TaxOrchestrator

    HAS_GPU = torch.cuda.is_available()
    if not HAS_GPU:
        error_msg = "torch.cuda.is_available() returned False. Ensure you are running in an environment with GPU support (e.g., WSL with CUDA)."
except Exception as e:
    HAS_GPU = False
    error_msg = str(e)


# Cache Model Loader (Prevents reloading model repeatedly in Streamlit)
@st.cache_resource
def load_model():
    if HAS_GPU:
        lora_path = os.path.join(
            os.path.dirname(__file__),
            "fine_tuning",
            "results_guardrails",
            "final_model",
        )
        if os.path.exists(lora_path):
            return ModelInferencer(lora_path=lora_path)
    return None


# Application Title
st.title("🤖 AI Payroll Tax Expert (SLM 7B)")
st.markdown(
    "*A Production-Grade Small Language Model Fine-Tuned for Indonesian Tax Unit Testing*"
)

# Navigation Tabs
tab_chat, tab_metodologi, tab_evaluasi, tab_konklusi = st.tabs(
    [
        "💬 Live Demo",
        "🧪 Research Methodology",
        "📊 Benchmark Results",
        "🏆 Key Conclusions",
    ]
)

# ==========================================
# TAB 1: LIVE DEMO (CHAT INTERFACE)
# ==========================================
with tab_chat:
    st.header("Direct Model Interaction")

    if HAS_GPU:
        st.success("✅ GPU detected. Model loaded locally.")
        model = load_model()
    else:
        st.warning(
            f"⚠️ Demo mode running in simulation (Mock Inference). Reason for GPU model loading failure: {error_msg}"
        )
        model = None

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Contoh: Buatkan unit test PPh21 untuk gaji 60 juta..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("Model is thinking..."):
                if model:
                    try:
                        # Invoke orchestrator pipeline
                        orchestrator = TaxOrchestrator()
                        full_response = orchestrator.process_chat(prompt, model)
                    except Exception as e:
                        full_response = (
                            f"An error occurred while processing the model: {e}"
                        )
                else:
                    # Fallback simulation for non-GPU hosting
                    time.sleep(2)
                    full_response = f"""
```python
def hitung_pajak(gaji):
    # (Peringatan: Berjalan tanpa GPU - Hanya menampilkan struktur fungsi)
    pass
```
*Note: Anda menjalankan Arsitektur **{selected_c.split(':')[0]}** tanpa GPU. Silakan jalankan secara lokal menggunakan GPU untuk mendapatkan output inference asli dari Qwen2.5!*
"""

            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

# ==========================================
# TAB 2: RESEARCH METHODOLOGY
# ==========================================
with tab_metodologi:
    st.header("Research Methodology")
    st.markdown("""
    This project adopts industry-standard Silicon Valley-grade evaluation methodology to demonstrate the feasibility of 7B Parameter Small Language Models (SLMs) in a domain-specific context (Indonesian Tax Law).
    
    ### Model Architecture
    - **Base Model:** Qwen2.5-7B-Instruct
    - **Training Method:** LoRA (Low-Rank Adaptation) with 4-bit Quantization (QLoRA)
    - **Dataset:** Synthetic PPh21 Dataset & Guardrails Data Blending
    
    ### Training Phases
    1. **Phase 1 (Domain Injection):** Rigorous training on tax mathematical rules and *Pytest* syntax.
    2. **Phase 2 (Guardrails & Alignment):** Data blending injection comprising 50% Tax, 25% Chit-Chat, and 25% Adversarial Rejection to mitigate Catastrophic Forgetting.
    """)

# ==========================================
# TAB 3: BENCHMARK RESULTS
# ==========================================
with tab_evaluasi:
    st.header("Evaluation & Visualization")

    st.subheader("1. Grokking Phenomenon & Learning Curve")
    st.markdown(
        "Below is empirical evidence of the model's intelligence evolution. The pure base model achieved only **9.2%** accuracy. However, by Step 396, accuracy surged to **100%**, demonstrating the **Grokking** phenomenon."
    )

    curve_path = os.path.join(os.path.dirname(__file__), "learning_curve_test.png")
    if os.path.exists(curve_path):
        st.image(Image.open(curve_path), caption="Learning Curve: Grokking Phenomenon")
    else:
        st.info("Learning curve chart is not yet available in the directory.")

    st.subheader("2. Baseline Ablation Study")
    st.markdown("""
    - **Base Model Score:** 9.2% (Pass@1)
    - **Final Model Score (Phase 2):** 96.1% (Pass@1)
    - **Delta Improvement:** ~86.9% absolute increase through Fine-Tuning.
    """)

    st.subheader("3. Hardware Profiling")
    st.markdown("""
    - **Peak VRAM:** 5.24 GB
    - **Inference Speed:** 10.74 tokens/sec
    - **Compute Conclusion:** Extremely lightweight and production-ready for deployment on on-premise enterprise servers without expensive GPU infrastructure.
    """)

# ==========================================
# TAB 4: CONCLUSION & FINDINGS (BLIND TEST)
# ==========================================
with tab_konklusi:
    st.header("The Real Blind Test & Fundamental SLM Limitations")

    st.markdown("""
    As a final validation, we provided an extreme prompt written in Indonesian colloquial slang without explicitly stating the tax formulas (*Tarif Efektif Rata-rata / TER*):
    
    > *"Bang, gaji bulanan gue 7.5 juta. Status gue belum nikah... tolong buatin script python ngitung potongan PPh21 pakai aturan TER dong."*
    
    **Key Scientific Finding:**
    The model successfully abstracted the *Pytest* syntax elegantly, but **COMPLETELY FAILED (Hallucinated)** regarding the PTKP regulatory thresholds and TER rules. 
    """)

    st.error("🚨 Key Finding: Fine-Tuning is NOT a Knowledge Base!")
    st.markdown("""
    1. **The Illusion of Expertise Transfer:** 100% accuracy on benchmarks merely demonstrates that the SLM successfully **memorized syntactic patterns** flawlessly, rather than internalizing the statutory tax law.
    2. **The Necessity of RAG (Retrieval-Augmented Generation):** This research concludes that to build a genuinely reliable legal/tax AI expert, the SLM must be paired with a RAG architecture so that regulatory logic is injected in real-time.
    
    *This research successfully delineates the architectural boundaries of 7B SLMs in modern software engineering.*
    """)
