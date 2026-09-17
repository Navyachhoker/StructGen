"""
Local inference service for the Qwen2.5-1.5B LoRA invoice extractor.

Runs separately from the main StructGen application so that
ML dependencies remain isolated in .venv-ml.
"""

import json
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ADAPTER_PATH = (
    PROJECT_ROOT
    / "content"
    / "qwen-invoice-lora-v3-recovered"
)


app = FastAPI(title="StructGen LoRA Inference Service")


class ExtractionRequest(BaseModel):
    invoice_text: str


class ExtractionResponse(BaseModel):
    output: str


def build_prompt(invoice_text: str) -> str:
    return f"""### Instruction:
You are an expert invoice data extraction system.

Extract the invoice/receipt text into JSON.

Required output fields:
- vendor_name
- invoice_number
- invoice_date
- line_items
- subtotal
- tax_amount
- total_amount
- currency

Each line item contains:
- description
- quantity
- unit_price
- total

Rules:
- Return ONLY valid JSON. No markdown or explanation.
- Do not guess or invent missing values.
- Dates must use YYYY-MM-DD format.
- Monetary values must be plain numbers as strings.

Invoice text:
\"\"\"
{invoice_text}
\"\"\"

JSON output:
"""


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    local_files_only=True,
)

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
)

model.eval()

print("LoRA model loaded successfully.")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "qwen-invoice-lora-v3",
    }


@app.post("/extract", response_model=ExtractionResponse)
def extract(request: ExtractionRequest) -> ExtractionResponse:
    prompt = build_prompt(request.invoice_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[0][input_length:]

    output_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # Make sure the service returns clean JSON text.
    output_text = output_text.strip()

    try:
        parsed = json.loads(output_text)
        output_text = json.dumps(parsed)
    except json.JSONDecodeError:
        # Return the raw model output.
        # Main application will perform schema validation.
        pass

    return ExtractionResponse(output=output_text)