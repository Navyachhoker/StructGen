"""
Builds the extraction prompt sent to a model.

Kept as a single function (not a class) since prompt construction is
stateless — it takes raw invoice text and returns a string. If the
project later needs multiple prompt strategies (e.g., a different
prompt for the fine-tuned model in Phase 7), add a second function here
rather than complicating this one with conditionals.
"""

from invoice_extractor.schemas.invoice import ExtractedInvoice

# Generated once from the schema so the prompt and schema can never
# silently drift apart from each other.
_SCHEMA_JSON = ExtractedInvoice.model_json_schema()


def build_extraction_prompt(raw_invoice_text: str) -> str:
    """
    Constructs a prompt instructing the model to extract structured
    invoice data as JSON matching our schema.

    The instruction to return ONLY JSON (no prose, no markdown fences)
    matters a lot in practice — models love to wrap JSON in ```json
    blocks or add explanatory text, which breaks naive parsing.
    """
    return f"""You are an expert invoice data extraction system.

Extract the following invoice/receipt text into JSON matching this exact schema:
{_SCHEMA_JSON}

Rules:
- Return ONLY valid JSON. No markdown code fences, no explanation, no extra text.
- If a field is not present in the text, omit it or use null — do not guess or invent values.
- Dates must be in YYYY-MM-DD format.
- Monetary values must be plain numbers as strings (no currency symbols).

Invoice text:
\"\"\"
{raw_invoice_text}
\"\"\"

JSON output:"""