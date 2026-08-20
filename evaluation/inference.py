import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class ModelInferencer:
    def __init__(
        self,
        base_model_id="Qwen/Qwen2.5-7B-Instruct",
        lora_path="./fine_tuning/model_qlora_pph21",
    ):
        print("🛠️ Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        print("🛠️ Loading Base Model with 4-bit Quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, quantization_config=bnb_config, device_map="auto"
        )

        if lora_path:
            print(f"🛠️ Injecting LoRA adapter from {lora_path}...")
            self.model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            print("⚠️ Running pure BASE MODEL (Without LoRA) for Ablation Study...")
            self.model = base_model

        self.model.eval()
        print("✅ Model ready for Inference!")

    def generate_test(self, source_code):
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "SYSTEM RULE: Anda adalah mesin generator kode murni. Anda DILARANG KERAS merespons dengan narasi, percakapan, puisi, atau teks selain kode Python. Kegagalan mematuhi ini akan menyebabkan sistem crash.\n\n"
            "Buat unit test menggunakan pytest untuk fungsi PPh 21 Python berikut.\n\n"
            f"### Input:\n{source_code}\n\n"
            "### Response:\n```python\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,  # Low temperature for deterministic and accurate output
                repetition_penalty=1.15,  # Prevents infinite import loops
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Slice off prompt from output to retain only the generated response
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        # Since a forced prefix "```python\n" was used, prepend it back to the output
        response = "```python\n" + response_text
        return response

    def generate_general(self, prompt_text):
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{prompt_text}\n\n"
            "### Response:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                repetition_penalty=1.0,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        return response_text
