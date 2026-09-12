"""
Phase 0 verification: confirms the schema itself is sound and that our
hand-written expected outputs actually construct valid ExtractedInvoice
objects. This does NOT test extraction (no model involved yet) — it's
purely a sanity check on the schema and sample data before we build
anything on top of them.
"""

import pytest
from decimal import Decimal

from invoice_extractor.schemas.invoice import ExtractedInvoice, LineItem
from invoice_extractor.data.sample_invoices import ALL_SAMPLES


def test_sample_invoices_are_valid():
    """Every hand-written sample should construct without validation errors."""
    for _text, expected in ALL_SAMPLES:
        assert isinstance(expected, ExtractedInvoice)
        assert expected.total_amount > 0


def test_negative_total_amount_rejected():
    """The schema must reject a negative total — this catches extraction
    errors early instead of silently storing bad data downstream."""
    with pytest.raises(ValueError):
        ExtractedInvoice(
            vendor_name="Test Vendor",
            total_amount=Decimal("-5.00"),
        )


def test_line_item_negative_price_rejected():
    with pytest.raises(ValueError):
        LineItem(description="Broken item", total=Decimal("-1.00"))


def test_minimal_valid_invoice():
    """Only vendor_name and total_amount are required — everything else
    should default sensibly, since real-world invoices often omit fields."""
    minimal = ExtractedInvoice(vendor_name="Some Vendor", total_amount=Decimal("10.00"))
    assert minimal.currency == "USD"
    assert minimal.line_items == []
    assert minimal.invoice_number is None