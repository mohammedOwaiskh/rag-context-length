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
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        max_new_tokens = self.generation_config.get("max_new_tokens") or 64
        repetition_penalty = self.generation_config.get("repetition_penalty") or 1.0

        start = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.generation_config.get("do_sample", False),
                temperature=None,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - start

        input_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip(), elapsed

    def generate_batch(self, prompts: list[str]) -> list[tuple[str, float]]:
        """
        Batched version of generate(). Returns one (text, time) tuple per
        prompt, in the same order as the input. The reported time per item
        is the shared batch time divided evenly — not each item's true
        individual cost, but sufficient for the reported generation_time
        field, since the whole point of batching is that items share cost.
        """
        batch_messages = [[{"role": "user", "content": p}] for p in prompts]
        inputs = self.tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,   # pad shorter prompts in the batch to the longest
        ).to(self.model.device)

        max_new_tokens = self.generation_config.get("max_new_tokens") or 64
        repetition_penalty = self.generation_config.get("repetition_penalty") or 1.0

        start = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.generation_config.get("do_sample", False),
                temperature=None,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elapsed = time.time() - start
        per_item_time = elapsed / len(prompts)

        input_len = inputs["input_ids"].shape[-1]  # same for every item, due to left-padding
        results = []
        for i in range(len(prompts)):
            new_tokens = output_ids[i][input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append((text.strip(), per_item_time))
        return results


class LlamaCppGGUFBackend:
    def __init__(self, model_name: str, generation_config: dict):
        raise NotImplementedError(
            "llama_cpp_gguf backend not yet implemented. To be filled in when "
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