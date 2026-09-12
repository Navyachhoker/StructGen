"""
Runs the hand-written sample invoices (Phase 0) through the real Groq
client and prints a results table.

This is a script, not a test — it costs real (tiny) API money and hits
a live network, so it doesn't belong in the automated pytest suite.
Run it manually whenever you want to check baseline behavior.
"""

from invoice_extractor.clients.groq_client import GroqClient
from invoice_extractor.data.sample_invoices import ALL_SAMPLES
from invoice_extractor.extraction import extract_invoice


def main():
    client = GroqClient()

    print(f"{'Sample':<10} {'Result':<10} {'Latency (s)':<12} {'Est. Cost ($)':<15} {'Notes'}")
    print("-" * 80)

    for i, (raw_text, _expected) in enumerate(ALL_SAMPLES, start=1):
        result = extract_invoice(raw_text, client)
        usage = client.get_last_usage()

        status = "PASS" if result.success else "FAIL"
        notes = "" if result.success else (result.error or "")[:40]

        print(
            f"invoice_{i:<3} {status:<10} {usage['latency_seconds']:<12} "
            f"{usage['estimated_cost_usd']:<15} {notes}"
        )


if __name__ == "__main__":
    main()