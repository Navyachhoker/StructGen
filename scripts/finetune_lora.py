import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter


# ============================================================
# Files
# ============================================================

RESULT_FILES = {
    "Groq — Synthetic": "/content/groq_synthetic_results.json",
    "Groq — Real": "/content/groq_real_results.json",
    "LoRA Qwen v3 — Real": "/content/lora_v3_synthetic_results.json",
    "Base Qwen — Real": "/content/base_qwen_real_results.json",
}


DOC_FIELDS = [
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "subtotal",
    "tax_amount",
    "total_amount",
    "currency",
]

ITEM_FIELDS = [
    "description",
    "quantity",
    "unit_price",
    "total",
]


# ============================================================
# Normalization
# ============================================================

def normalize_text(value):
    if value is None:
        return None

    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)

    text = text.replace("–", "-").replace("—", "-")

    text = re.sub(
        r"\s*([,/+#*@().-])\s*",
        r"\1",
        text
    )

    return text


def normalize_number(value):
    if value is None:
        return None

    try:
        text = str(value).strip().replace(",", "")

        if text == "":
            return None

        return float(text)

    except (ValueError, TypeError):
        return None


def normalize_currency(value):
    if value is None:
        return None

    value = str(value).strip().upper()

    aliases = {
        "RM": "MYR",
        "MYR": "MYR",
        "RINGGIT": "MYR",
        "MALAYSIAN RINGGIT": "MYR",
    }

    return aliases.get(value, value)


def numbers_equal(a, b, tolerance=1e-6):

    a_num = normalize_number(a)
    b_num = normalize_number(b)

    if a_num is None or b_num is None:
        return a_num is None and b_num is None

    return abs(a_num - b_num) <= tolerance


def field_equal(field, gt, pred):

    if field == "description":
        return normalize_text(gt) == normalize_text(pred)

    if field in {
        "quantity",
        "unit_price",
        "total",
        "subtotal",
        "tax_amount",
    }:
        return numbers_equal(gt, pred)

    if field == "currency":
        return normalize_currency(gt) == normalize_currency(pred)

    return gt == pred


# ============================================================
# Line-item matching
# ============================================================

def description_similarity(a, b):

    a = normalize_text(a)
    b = normalize_text(b)

    if a is None or b is None:
        return 0.0

    if a == b:
        return 1.0

    return SequenceMatcher(None, a, b).ratio()


def item_score(gt, pred):

    desc = description_similarity(
        gt.get("description"),
        pred.get("description")
    )

    quantity = numbers_equal(
        gt.get("quantity"),
        pred.get("quantity")
    )

    unit_price = numbers_equal(
        gt.get("unit_price"),
        pred.get("unit_price")
    )

    total = numbers_equal(
        gt.get("total"),
        pred.get("total")
    )

    return (
        0.55 * desc
        + 0.15 * quantity
        + 0.15 * unit_price
        + 0.15 * total
    )


def items_match(gt, pred):

    desc = description_similarity(
        gt.get("description"),
        pred.get("description")
    )

    quantity = numbers_equal(
        gt.get("quantity"),
        pred.get("quantity")
    )

    unit_price = numbers_equal(
        gt.get("unit_price"),
        pred.get("unit_price")
    )

    total = numbers_equal(
        gt.get("total"),
        pred.get("total")
    )

    if desc == 1.0:
        return True

    if desc >= 0.90:
        return True

    if desc >= 0.75 and (total or unit_price):
        return True

    if desc >= 0.60 and total and (quantity or unit_price):
        return True

    return False


def match_items(gt_items, pred_items):

    remaining = set(range(len(pred_items)))

    matches = []
    missing = []

    for gt_index, gt_item in enumerate(gt_items):

        candidates = []

        for pred_index in remaining:

            pred_item = pred_items[pred_index]

            if items_match(gt_item, pred_item):

                score = item_score(
                    gt_item,
                    pred_item
                )

                candidates.append(
                    (score, pred_index)
                )

        if not candidates:
            missing.append(gt_index)
            continue

        candidates.sort(reverse=True)

        _, best_pred_index = candidates[0]

        matches.append(
            (gt_index, best_pred_index)
        )

        remaining.remove(best_pred_index)

    extra = sorted(remaining)

    return matches, missing, extra


# ============================================================
# Evaluation
# ============================================================

def evaluate(file_path):

    data = json.loads(
        Path(file_path).read_text(
            encoding="utf-8"
        )
    )

    results = data.get("results", [])

    total_examples = len(results)

    valid_json = 0

    doc_correct = Counter()
    doc_total = Counter()

    item_correct = Counter()
    item_total = Counter()

    hallucinations = Counter()

    gt_item_count = 0
    pred_item_count = 0
    matched_item_count = 0
    missing_item_count = 0
    extra_item_count = 0

    complete_items = 0

    example_details = []

    for result in results:

        gt = result.get("ground_truth") or {}
        pred = result.get("prediction")

        if isinstance(pred, dict):
            valid_json += 1
        else:
            pred = {}

        # ----------------------------------------------------
        # Document-level fields
        # ----------------------------------------------------

        for field in DOC_FIELDS:

            gt_value = gt.get(field, None)
            pred_value = pred.get(field, None)

            doc_total[field] += 1

            if field_equal(
                field,
                gt_value,
                pred_value
            ):
                doc_correct[field] += 1

            # Ground truth is missing but model produced value
            if (
                gt_value is None
                and pred_value is not None
            ):
                hallucinations[field] += 1

        # ----------------------------------------------------
        # Line items
        # ----------------------------------------------------

        gt_items = gt.get("line_items") or []
        pred_items = pred.get("line_items") or []

        gt_item_count += len(gt_items)
        pred_item_count += len(pred_items)

        matches, missing, extra = match_items(
            gt_items,
            pred_items
        )

        matched_item_count += len(matches)
        missing_item_count += len(missing)
        extra_item_count += len(extra)

        for gt_index, pred_index in matches:

            gt_item = gt_items[gt_index]
            pred_item = pred_items[pred_index]

            all_fields_match = True

            for field in ITEM_FIELDS:

                gt_value = gt_item.get(field, None)
                pred_value = pred_item.get(field, None)

                item_total[field] += 1

                if field_equal(
                    field,
                    gt_value,
                    pred_value
                ):
                    item_correct[field] += 1
                else:
                    all_fields_match = False

                if (
                    gt_value is None
                    and pred_value is not None
                ):
                    hallucinations[
                        f"line_item.{field}"
                    ] += 1

            if all_fields_match:
                complete_items += 1

        example_details.append({
            "index": result.get("index"),
            "gt_items": len(gt_items),
            "pred_items": len(pred_items),
            "matched": len(matches),
            "missing": len(missing),
            "extra": len(extra),
            "missing_indices": missing,
            "extra_indices": extra,
        })

    # ========================================================
    # Metrics
    # ========================================================

    def percentage(correct, total):

        if total == 0:
            return 0.0

        return (
            100 * correct / total
        )

    doc_accuracy = {
        field: percentage(
            doc_correct[field],
            doc_total[field]
        )
        for field in DOC_FIELDS
    }

    total_doc_correct = sum(
        doc_correct.values()
    )

    total_doc_fields = sum(
        doc_total.values()
    )

    overall_field_accuracy = percentage(
        total_doc_correct,
        total_doc_fields
    )

    item_field_accuracy = {
        field: percentage(
            item_correct[field],
            item_total[field]
        )
        for field in ITEM_FIELDS
    }

    item_recall = percentage(
        matched_item_count,
        gt_item_count
    )

    item_precision = percentage(
        matched_item_count,
        pred_item_count
    )

    complete_item_accuracy = percentage(
        complete_items,
        gt_item_count
    )

    return {
        "examples": total_examples,

        "valid_json": valid_json,

        "valid_json_rate": percentage(
            valid_json,
            total_examples
        ),

        "overall_field_accuracy":
            overall_field_accuracy,

        "doc_correct": dict(doc_correct),
        "doc_total": dict(doc_total),
        "doc_accuracy": doc_accuracy,

        "gt_items": gt_item_count,
        "predicted_items": pred_item_count,
        "matched_items": matched_item_count,
        "missing_items": missing_item_count,
        "extra_items": extra_item_count,

        "item_recall": item_recall,
        "item_precision": item_precision,

        "item_correct": dict(item_correct),
        "item_total": dict(item_total),
        "item_field_accuracy":
            item_field_accuracy,

        "complete_items": complete_items,
        "complete_item_accuracy":
            complete_item_accuracy,

        "hallucinations":
            dict(hallucinations),

        "example_details":
            example_details,
    }


# ============================================================
# Run benchmark
# ============================================================

all_metrics = {}

for model_name, file_path in RESULT_FILES.items():

    print("\n" + "=" * 90)
    print(model_name)
    print("=" * 90)

    try:

        metrics = evaluate(file_path)

        all_metrics[model_name] = metrics

        print(
            f"Examples:              "
            f"{metrics['examples']}"
        )

        print(
            f"Valid JSON:            "
            f"{metrics['valid_json']}/"
            f"{metrics['examples']} "
            f"({metrics['valid_json_rate']:.2f}%)"
        )

        print(
            f"Overall field accuracy: "
            f"{metrics['overall_field_accuracy']:.2f}%"
        )

        print("\nDocument-level fields:")

        for field in DOC_FIELDS:

            correct = metrics[
                "doc_correct"
            ][field]

            total = metrics[
                "doc_total"
            ][field]

            accuracy = metrics[
                "doc_accuracy"
            ][field]

            print(
                f"  {field:<20}"
                f"{correct:>3}/{total:<3} "
                f"({accuracy:>6.2f}%)"
            )

        print("\nLine items:")

        print(
            f"  Ground-truth items:   "
            f"{metrics['gt_items']}"
        )

        print(
            f"  Predicted items:      "
            f"{metrics['predicted_items']}"
        )

        print(
            f"  Matched items:        "
            f"{metrics['matched_items']}"
        )

        print(
            f"  Missing items:        "
            f"{metrics['missing_items']}"
        )

        print(
            f"  Extra items:          "
            f"{metrics['extra_items']}"
        )

        print(
            f"  Item recall:          "
            f"{metrics['item_recall']:.2f}%"
        )

        print(
            f"  Item precision:       "
            f"{metrics['item_precision']:.2f}%"
        )

        print(
            f"  Complete item acc.:   "
            f"{metrics['complete_item_accuracy']:.2f}%"
        )

        print("\nLine-item field accuracy:")

        for field in ITEM_FIELDS:

            correct = metrics[
                "item_correct"
            ][field]

            total = metrics[
                "item_total"
            ][field]

            accuracy = metrics[
                "item_field_accuracy"
            ][field]

            print(
                f"  {field:<20}"
                f"{correct:>3}/{total:<3} "
                f"({accuracy:>6.2f}%)"
            )

        if metrics["hallucinations"]:

            print("\nHallucinated values:")

            for field, count in (
                metrics["hallucinations"].items()
            ):

                print(
                    f"  {field:<25}{count}"
                )

    except Exception as e:

        print(
            f"ERROR evaluating "
            f"{model_name}: {e}"
        )


# ============================================================
# Benchmark summary
# ============================================================

print("\n\n")
print("=" * 115)
print("BENCHMARK SUMMARY")
print("=" * 115)

print(
    f"{'Model / Dataset':<28}"
    f"{'Valid JSON':>12}"
    f"{'Field Acc.':>12}"
    f"{'Item Recall':>13}"
    f"{'Item Precision':>15}"
    f"{'Complete Item':>15}"
)

print("-" * 115)

for model_name, metrics in all_metrics.items():

    print(
        f"{model_name:<28}"
        f"{metrics['valid_json_rate']:>11.2f}%"
        f"{metrics['overall_field_accuracy']:>11.2f}%"
        f"{metrics['item_recall']:>12.2f}%"
        f"{metrics['item_precision']:>14.2f}%"
        f"{metrics['complete_item_accuracy']:>14.2f}%"
    )


# ============================================================
# Save final benchmark
# ============================================================

output_path = "/content/corrected_benchmark_results.json"

Path(output_path).write_text(
    json.dumps(
        {
            name: {
                key: value
                for key, value in metrics.items()
                if key != "example_details"
            }
            for name, metrics in all_metrics.items()
        },
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

print("\nSaved:")
print(output_path)