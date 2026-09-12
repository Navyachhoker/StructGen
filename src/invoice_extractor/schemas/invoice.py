"""
The extraction contract.

This Pydantic model defines exactly what fields a "correctly extracted
invoice" must contain. Both the Groq baseline and the fine-tuned small
model will have their JSON output validated against this schema — a
model's output that doesn't parse into this structure is treated as a
validation failure and triggers the router's fallback logic (Phase 3+).

Keep this schema stable once fine-tuning (Phase 7) begins — the small
model is trained to predict exactly this shape.
"""

from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LineItem(BaseModel):
    """A single line item on an invoice (e.g., one product/service billed)."""

    description: str = Field(..., description="Text description of the item or service")
    quantity: Optional[float] = Field(default=None, description="Units purchased, if stated")
    unit_price: Optional[Decimal] = Field(default=None, description="Price per unit, if stated")
    total: Decimal = Field(..., description="Total price for this line item")

    @field_validator("total", "unit_price")
    @classmethod
    def non_negative(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Prices should never be negative — catches obvious extraction errors early."""
        if v is not None and v < 0:
            raise ValueError("Monetary values must be non-negative")
        return v


class ExtractedInvoice(BaseModel):
    """
    The full structured representation of an invoice or receipt.

    Fields are deliberately Optional where real-world documents commonly
    omit them (e.g., a handwritten receipt might not have an invoice
    number). Required fields are the ones a downstream bookkeeping
    system realistically cannot function without.
    """

    vendor_name: str = Field(..., description="Name of the business that issued the invoice")
    invoice_number: Optional[str] = Field(default=None, description="Invoice/receipt ID, if present")
    invoice_date: Optional[date] = Field(default=None, description="Date the invoice was issued")

    line_items: list[LineItem] = Field(
        default_factory=list,
        description="Itemized list of goods/services billed",
    )

    subtotal: Optional[Decimal] = Field(default=None, description="Total before tax")
    tax_amount: Optional[Decimal] = Field(default=None, description="Tax charged, if stated")
    total_amount: Decimal = Field(..., description="Final amount due — the one field we treat as mandatory")

    currency: str = Field(default="USD", description="ISO currency code, defaults to USD if not stated")

    @field_validator("total_amount", "subtotal", "tax_amount")
    @classmethod
    def non_negative(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        if v is not None and v < 0:
            raise ValueError("Monetary values must be non-negative")
        return v

    class Config:
        # Allows Decimal <-> JSON-string round-tripping cleanly,
        # which matters once this gets serialized for Postgres storage (Phase 5).
        json_encoders = {Decimal: str}