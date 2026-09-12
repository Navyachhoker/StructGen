"""
Orchestrates the full extraction pipeline: prompt -> model call -> parse -> validate.

This is the one place that ties together prompts.py, a ModelClient, and
the schema. Both the baseline (Groq) and fine-tuned (Ollama) paths reuse
this exact function later — only the client passed in changes.
"""

import json
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from invoice_extractor.clients.base import ModelClient
from invoice_extractor.prompts import build_extraction_prompt
from invoice_extractor.schemas.invoice import ExtractedInvoice


@dataclass
class ExtractionResult:
    """
    Wraps the outcome of one extraction attempt.

    Carrying both success and failure info in one object (rather than
    raising exceptions) matters for Phase 3+: the router needs to
    inspect *why* an attempt failed (bad JSON vs. schema mismatch) to
    decide whether to fall back to the baseline model, without the
    caller needing a try/except around every call.
    """

    success: bool
    invoice: Optional[ExtractedInvoice] = None
    raw_output: str = ""
    error: Optional[str] = None


def extract_invoice(raw_text: str, client: ModelClient) -> ExtractionResult:
    """
    Runs one full extraction attempt against the given client.

    Any client implementing ModelClient works here unchanged — this
    function has no knowledge of Groq, Ollama, or fakes.
    """
    prompt = build_extraction_prompt(raw_text)
    raw_output = client.extract_raw(prompt)

    # Step 1: parse JSON. Models occasionally return malformed JSON —
    # this is treated as a failure, not an exception that crashes the caller.
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as e:
        return ExtractionResult(success=False, raw_output=raw_output, error=f"Invalid JSON: {e}")

    # Step 2: validate against schema. Pydantic does the actual field-level
    # checking (types, required fields, our custom non-negative validators) —
    # we don't reimplement any of that here.
    try:
        invoice = ExtractedInvoice(**parsed)
    except ValidationError as e:
        return ExtractionResult(success=False, raw_output=raw_output, error=f"Schema validation failed: {e}")

    return ExtractionResult(success=True, invoice=invoice, raw_output=raw_output)