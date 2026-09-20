import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class HFBnbBackend:
    def __init__(self, model_name: str, generation_config: dict):
        print(f"Loading {model_name} in 4-bit (nf4) via bitsandbytes...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model.eval()
        self.generation_config = generation_config

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, prompt: str) -> tuple[str, float]:
        """Returns (generated_text, generation_time_seconds)."""
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        max_new_tokens = self.generation_config.get("max_new_tokens") or 64
        repetition_penalty = self.generation_config.get("repetition_penalty") or 1.0

        start = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=self.generation_config.get("do_sample", False),
                temperature=None,  # must be unset when do_sample=False, or HF warns/errors
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - start

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip(), elapsed


class LlamaCppGGUFBackend:
    def __init__(self, model_name: str, generation_config: dict):
        raise NotImplementedError(
            "llama_cpp_gguf backend not yet implemented. Fill this in when "
            "moving to the CPU-only path — same generate() interface as "
            "HFBnbBackend so the rest of the pipeline (generate.py, "
            "run_pilot.py) requires no changes."
        )


def load_backend(cfg: dict):
    backend_name = cfg["generation"]["backend"]
    model_name = cfg["generation"]["model"][backend_name]
    generation_config = cfg["generation"]["generation_config"]

    if backend_name == "hf_bnb":
        return HFBnbBackend(model_name, generation_config)
    elif backend_name == "llama_cpp_gguf":
        return LlamaCppGGUFBackend(model_name, generation_config)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")