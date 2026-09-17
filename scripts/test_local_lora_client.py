from invoice_extractor.clients.local_lora_client import LocalLoRAClient


def main() -> None:
    client = LocalLoRAClient()

    invoice_text = """TechMart
Invoice Number: INV-1001
Invoice Date: 2026-09-16

Laptop Stand    1    45.02    45.02
Mousepad        3    78.96    236.88
USB Hub         2    12.55    25.10

Subtotal: 306.00
Tax: 15.50
Total: 321.50
Currency: USD
"""

    raw_output = client.extract_raw(
        f"""Invoice text:
\"\"\"
{invoice_text}
\"\"\"

JSON output:
"""
    )

    print("\nLoRA client output:")
    print(raw_output)

    print("\nUsage:")
    print(client.get_last_usage())


if __name__ == "__main__":
    main()