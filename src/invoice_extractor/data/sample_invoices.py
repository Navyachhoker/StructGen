"""
Hand-written sample invoices + their correct expected extraction.

These are NOT for training the fine-tuned model — keep this set untouched
and use it exclusively as a held-out test set (see Phase 8). Mixing test
data into training data is one of the most common ways a benchmark ends
up dishonest.
"""

from decimal import Decimal
from datetime import date

from invoice_extractor.schemas.invoice import ExtractedInvoice, LineItem

# Raw text as it might realistically appear (OCR'd or plain text, with
# formatting quirks intentionally left in — real invoices are not clean).
SAMPLE_INVOICE_001_TEXT = """
ACME OFFICE SUPPLIES
Invoice #: INV-2024-0091
Date: 2024-03-15

Description                Qty     Unit Price      Total
Printer Paper (Box)         3        $24.99         $74.97
Stapler                     1        $8.50          $8.50

Subtotal:  $83.47
Tax (8%):  $6.68
Total Due: $90.15
"""

SAMPLE_INVOICE_001_EXPECTED = ExtractedInvoice(
    vendor_name="ACME Office Supplies",
    invoice_number="INV-2024-0091",
    invoice_date=date(2024, 3, 15),
    line_items=[
        LineItem(description="Printer Paper (Box)", quantity=3, unit_price=Decimal("24.99"), total=Decimal("74.97")),
        LineItem(description="Stapler", quantity=1, unit_price=Decimal("8.50"), total=Decimal("8.50")),
    ],
    subtotal=Decimal("83.47"),
    tax_amount=Decimal("6.68"),
    total_amount=Decimal("90.15"),
    currency="USD",
)

# A messier example — no invoice number, informal formatting.
# Good for stress-testing extraction logic in Phase 2.
SAMPLE_INVOICE_002_TEXT = """
Joe's Coffee Cart
3/22 - thanks for stopping by!

2x Latte ....... 9.00
1x Bagel ........ 3.50

total: $12.50 (cash)
"""

SAMPLE_INVOICE_002_EXPECTED = ExtractedInvoice(
    vendor_name="Joe's Coffee Cart",
    invoice_number=None,
    invoice_date=date(2024, 3, 22),
    line_items=[
        LineItem(description="Latte", quantity=2, unit_price=None, total=Decimal("9.00")),
        LineItem(description="Bagel", quantity=1, unit_price=None, total=Decimal("3.50")),
    ],
    subtotal=None,
    tax_amount=None,
    total_amount=Decimal("12.50"),
    currency="USD",
)

# Collect all samples for easy iteration in tests / later benchmarking.
ALL_SAMPLES = [
    (SAMPLE_INVOICE_001_TEXT, SAMPLE_INVOICE_001_EXPECTED),
    (SAMPLE_INVOICE_002_TEXT, SAMPLE_INVOICE_002_EXPECTED),
]