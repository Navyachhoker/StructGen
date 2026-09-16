from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "content"
    / "qwen-invoice-lora-v3-recovered"
)


def main() -> None:
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH,
        local_files_only=True,
    )

    print("Loading base model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
    )

    model.eval()

    print("Model loaded successfully.")
    print(f"Base model: {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Device: {model.device}")


if __name__ == "__main__":
    main()