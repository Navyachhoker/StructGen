"""
Request/response models for the HTTP API layer.

These are deliberately separate from ExtractedInvoice (schemas/invoice.py).
That schema is the *extraction contract* the model output is judged
against. These models are the *API contract* — what a client sends/
receives over HTTP. Conflating the two would make it harder to evolve
one without breaking the other (e.g., adding a request field like
`preferred_model` here shouldn't touch the extraction schema at all).
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel

from invoice_extractor.schemas.invoice import ExtractedInvoice


class ExtractRequest(BaseModel):
    """Body for POST /extract."""

    raw_text: str


class ExtractResponse(BaseModel):
    """Body returned by POST /extract.

    `model_name` and `fallback_used` are populated by worker.py's
    extract_invoice_task (see the result_payload dict there) and read
    by the Streamlit UI to show which model actually served a given
    job. Without declaring them here, Pydantic silently drops them
    when /jobs/{job_id} reconstructs this model from the arq job
    result, which is why the UI would otherwise always show
    "unknown" / "No fallback".
    """

    success: bool
    invoice: Optional[ExtractedInvoice] = None
    error: Optional[str] = None
    latency_seconds: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    model_name: Optional[str] = None
    fallback_used: bool = False


class StatsResponse(BaseModel):
    """Body returned by GET /stats."""

    total_requests: int
    successful_requests: int
    success_rate: float
    average_latency_seconds: Optional[float] = None
    total_estimated_cost_usd: float
    
class JobSubmitResponse(BaseModel):
    """Returned immediately by POST /extract, before any work is done."""

    job_id: str
    status: str = "queued"


class JobStatusResponse(BaseModel):
    """Returned by GET /jobs/{job_id}. `result` is populated only once status is 'complete'."""

    job_id: str
    status: str  # "deferred" | "queued" | "in_progress" | "complete" | "not_found"
    result: Optional[ExtractResponse] = None
    
    
    
class InvoiceListItem(BaseModel):
    job_id: str
    model_name: str
    success: bool
    error: Optional[str] = None
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    total_amount: Optional[str] = None
    latency_seconds: float
    estimated_cost_usd: float
    created_at: datetime