"""
Prints the specific line items flagged by analyze_synthetic_data.py as
"description missing from raw_text", so you can manually confirm they're
benign abbreviations/reformatting rather than actual value corruption.

Run from your StructGen project root:
    python inspect_flagged_descriptions.py
"""
import json
from pathlib import Path

PATH = Path("data/synthetic/raw_pairs.jsonl")
MAX_TO_SHOW = 15  # cap output length for readability


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
    shown = 0

    for p in pairs:
        raw = p["raw_text"]
        inv = p["invoice"]
        for item in inv["line_items"]:
            desc = item["description"]
            if desc.lower() not in raw.lower():
                if shown >= MAX_TO_SHOW:
                    print(f"... (stopping after {MAX_TO_SHOW}; there may be more)")
                    return
                print(f"--- Flagged description: {desc!r} ---")
                print(f"Ground truth line item: {item}")
                print(f"Rendered text:\n{raw}")
                print()
                shown += 1

    print(f"\nTotal flagged shown: {shown}")


if __name__ == "__main__":
    main()