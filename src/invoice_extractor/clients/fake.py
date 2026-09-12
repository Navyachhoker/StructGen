"""
A fake model client for testing the extraction pipeline without any
real API calls. Returns a fixed, valid JSON string regardless of input
by default, or a caller-supplied response for testing specific cases
(e.g., malformed JSON, to test failure handling).
"""

from invoice_extractor.clients.base import ModelClient

# A minimal but schema-valid response, used as the default canned output.
DEFAULT_FAKE_RESPONSE = """
{
  "vendor_name": "Fake Vendor Inc",
  "invoice_number": "FAKE-001",
  "invoice_date": "2024-01-01",
  "line_items": [
    {"description": "Sample Item", "quantity": 1, "unit_price": "10.00", "total": "10.00"}
  ],
  "subtotal": "10.00",
  "tax_amount": "0.00",
  "total_amount": "10.00",
  "currency": "USD"
}
"""


class FakeModelClient(ModelClient):
    """
    Test double for ModelClient. Pass a custom `fixed_response` to
    simulate specific model behavior (e.g., malformed output) without
    needing a real API.
    """

    def __init__(self, fixed_response: str = DEFAULT_FAKE_RESPONSE):
        self.fixed_response = fixed_response

    def extract_raw(self, prompt: str) -> str:
        # Ignores the prompt entirely — this client's only job is to
        # return whatever canned response the test asked for.
        return self.fixed_response