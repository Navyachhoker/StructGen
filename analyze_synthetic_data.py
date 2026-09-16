"""
Analyzes data/synthetic/raw_pairs.jsonl for diversity and quality signals.
Run from your StructGen project root:

    python analyze_synthetic_data.py

This does NOT call any LLM - it's pure local analysis of what you already
generated, so it's free and instant.
"""
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

PATH = Path("data/synthetic/raw_pairs.jsonl")


def load_pairs():
    pairs = []
    with open(PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def main():
    pairs = load_pairs()
    n = len(pairs)
    print(f"Total examples: {n}\n")

    # --- Diversity: vendor names ---
    vendors = Counter(p["invoice"]["vendor_name"] for p in pairs)
    print(f"Unique vendor names: {len(vendors)} (out of {n} examples)")
    print("Top 5 most common vendors:")
    for v, c in vendors.most_common(5):
        print(f"  {v}: {c}")
    print()

    # --- Diversity: currency spread ---
    currencies = Counter(p["invoice"]["currency"] for p in pairs)
    print("Currency distribution:")
    for cur, c in currencies.most_common():
        print(f"  {cur}: {c} ({c/n:.0%})")
    print()

    # --- Diversity: line item counts ---
    item_counts = [len(p["invoice"]["line_items"]) for p in pairs]
    print(f"Line items per invoice: min={min(item_counts)}, "
          f"max={max(item_counts)}, avg={sum(item_counts)/n:.1f}")
    print()

    # --- Field omission rates (are nulls actually varied, not all-or-nothing?) ---
    omit_invoice_num = sum(1 for p in pairs if p["invoice"]["invoice_number"] is None)
    omit_date = sum(1 for p in pairs if p["invoice"]["invoice_date"] is None)
    omit_subtotal = sum(1 for p in pairs if p["invoice"]["subtotal"] is None)
    omit_tax = sum(1 for p in pairs if p["invoice"]["tax_amount"] is None)
    print("Field omission rates (null in ground truth):")
    print(f"  invoice_number: {omit_invoice_num}/{n} ({omit_invoice_num/n:.0%})")
    print(f"  invoice_date:   {omit_date}/{n} ({omit_date/n:.0%})")
    print(f"  subtotal:       {omit_subtotal}/{n} ({omit_subtotal/n:.0%})")
    print(f"  tax_amount:     {omit_tax}/{n} ({omit_tax/n:.0%})")
    print()

    # --- Text length variety (proxy for style diversity - clean vs messy) ---
    lengths = [len(p["raw_text"]) for p in pairs]
    print(f"Rendered text length (chars): min={min(lengths)}, "
          f"max={max(lengths)}, avg={sum(lengths)/n:.0f}")
    print()

    # --- CRITICAL quality check: does the rendered text actually contain ---
    # --- every line item description and the total amount? ---
    missing_desc_count = 0
    missing_total_count = 0
    for p in pairs:
        raw = p["raw_text"]
        inv = p["invoice"]
        for item in inv["line_items"]:
            desc = item["description"]
            if desc.lower() not in raw.lower():
                missing_desc_count += 1
        if str(inv["total_amount"]) not in raw:
            missing_total_count += 1

    print("Value-fidelity check (rendered text actually contains ground truth):")
    print(f"  Line items whose description text is MISSING from raw_text: {missing_desc_count}")
    print(f"  Invoices where total_amount string is MISSING from raw_text: {missing_total_count}/{n}")
    print("  (Some false positives are expected here - e.g. abbreviated")
    print("   descriptions like 'Grld Chkn' won't match exactly. Treat this")
    print("   as a rough signal, not an exact defect count - spot check any")
    print("   flagged examples manually before worrying about them.)")
    print()

    # --- Duplicate / near-duplicate check ---
    exact_dupes = n - len(set(p["raw_text"] for p in pairs))
    print(f"Exact duplicate raw_text entries: {exact_dupes}")


if __name__ == "__main__":
    main()