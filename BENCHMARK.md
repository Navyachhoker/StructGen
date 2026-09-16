# StructGen — Benchmark & Evaluation

## 1. Overview

StructGen is an invoice and receipt information extraction system designed to convert semi-structured financial documents into a consistent structured representation.

The project benchmarks three extraction approaches:

1. **Base Qwen** — a base Qwen model evaluated without task-specific fine-tuning.
2. **Groq Prompted Baseline** — a larger instruction-tuned model accessed through the Groq API.
3. **LoRA Fine-Tuned Model** — a Qwen model adapted to the invoice extraction task using Low-Rank Adaptation (LoRA).

The primary objective is not only to measure extraction accuracy, but to investigate how well a relatively small fine-tuned model generalizes from synthetic training data to previously unseen real-world invoices and receipts.

---

# 2. Benchmark Objectives

The benchmark was designed to answer the following questions:

* How accurately can the models extract structured invoice information?
* How does a base model compare with a prompted larger model?
* Does task-specific LoRA fine-tuning improve structured extraction?
* How well does a model trained primarily on synthetic examples generalize to real documents?
* Which document fields are more difficult to extract reliably?
* What limitations arise when extracting line items from messy real-world receipts?
* Can a smaller fine-tuned model provide a practical alternative to a larger prompted model?

The benchmark therefore evaluates both **raw extraction performance** and **generalization to real-world documents**.

---

# 3. Extraction Schema

Each document is converted into a structured invoice representation.

The benchmark evaluates the following document-level fields:

| Field            | Description                   |
| ---------------- | ----------------------------- |
| `vendor`         | Vendor or merchant name       |
| `invoice_number` | Invoice or receipt identifier |
| `date`           | Invoice/transaction date      |
| `subtotal`       | Pre-tax subtotal              |
| `tax`            | Tax amount                    |
| `total`          | Final payable amount          |
| `currency`       | Currency used in the document |

Line items are evaluated separately.

A line item may contain fields such as:

| Field         | Description                    |
| ------------- | ------------------------------ |
| `description` | Product or service description |
| `quantity`    | Number of units                |
| `unit_price`  | Price per unit                 |
| `total`       | Total price for the line item  |

This separation is intentional because extracting document-level metadata and reconstructing variable-length line-item tables represent different challenges.

---

# 4. Dataset Design

The benchmark uses both **synthetic** and **real-world** documents.

## Synthetic Dataset

Synthetic invoices were created to provide controlled training and evaluation examples.

Synthetic documents allow variation in:

* vendor names
* invoice identifiers
* dates
* currencies
* tax values
* subtotals and totals
* number of line items
* formatting
* document layouts
* missing fields
* receipt-like structures

The synthetic dataset was used primarily for model development and evaluation.

## Real-World Holdout

A separate set of real invoices and receipts was used as an out-of-distribution evaluation set.

The purpose of this split is to measure whether a model trained on structured or synthetic examples can generalize to documents containing characteristics such as:

* irregular formatting
* noisy text
* receipt-style layouts
* inconsistent field placement
* varying numbers of line items
* unfamiliar vendors
* different conventions for prices, dates, taxes, and totals

The real-world holdout therefore provides a more realistic measure of deployment-oriented performance.

---

# 5. Models Compared

## 5.1 Base Qwen

The first baseline uses a base Qwen model without task-specific fine-tuning.

Its purpose is to establish how well the underlying model performs on the extraction task before adaptation.

This provides a reference point for determining whether subsequent prompting or fine-tuning provides measurable improvement.

---

## 5.2 Groq Prompted Baseline

The second baseline uses a larger instruction-following model through the Groq API.

The model receives the invoice text together with an extraction prompt and is instructed to produce output matching the StructGen schema.

This baseline represents a common production approach:

> Use a capable general-purpose LLM with carefully designed prompting rather than maintaining a task-specific fine-tuned model.

The Groq model therefore provides a useful comparison against the smaller fine-tuned model.

---

## 5.3 LoRA Fine-Tuned Model

The third approach uses LoRA to adapt the Qwen model specifically for structured invoice extraction.

LoRA updates a small set of trainable low-rank parameters rather than updating the entire base model.

The goal is to determine whether task-specific adaptation can improve structured extraction while keeping the model relatively small and computationally efficient.

The fine-tuning experiment was performed using a limited training budget, making the experiment representative of a constrained ML engineering environment rather than a large-scale training setup.

---

# 6. Evaluation Methodology

The benchmark evaluates predictions against the ground-truth structured representation.

## Document-Level Accuracy

For each document-level field, the predicted value is compared against the corresponding ground-truth value.

Conceptually:

```text
field accuracy =
correctly extracted fields
--------------------------
total evaluated fields
```

The benchmark aggregates these comparisons across the evaluation set.

This produces a field-level accuracy metric that indicates how reliably the model reconstructs the structured document representation.

---

# 7. Normalization

Raw model output cannot always be compared directly to ground truth.

The evaluation therefore considers normalized representations where appropriate.

Examples include:

* consistent treatment of whitespace
* normalization of textual casing
* normalization of numeric representations
* normalization of equivalent formatting

Normalization is important because formatting differences do not necessarily represent extraction errors.

For example:

```text
$1,250.00
1250.00
1250
```

may represent the same underlying numeric value depending on the schema and evaluator.

The benchmark distinguishes representation differences from actual semantic differences wherever the evaluator supports this distinction.

---

# 8. Line-Item Evaluation

Line items are evaluated independently from document-level fields.

This is necessary because line-item extraction introduces additional challenges:

* variable numbers of items
* ordering differences
* missing items
* extra predicted items
* incorrect descriptions
* incorrect quantities
* incorrect prices
* incorrect totals

A model can therefore achieve reasonable document-field accuracy while still performing poorly at reconstructing the complete line-item table.

For this reason, line-item performance should not be inferred solely from the overall document-field accuracy.

---

# 9. Real-World Benchmark Results

The real-world holdout was used to evaluate how the models generalize beyond the synthetic training distribution.

The currently verified LoRA result is:

| Model                | Dataset            | Field Accuracy |
| -------------------- | ------------------ | -------------: |
| LoRA Fine-Tuned Qwen | Real-world holdout |     **54.95%** |

The intermediate JSON result files containing the complete per-example predictions were subsequently lost. Therefore, the detailed per-field and per-example breakdown is not included here unless the original outputs are recovered.

This distinction is important: the benchmark records the verified aggregate result rather than reconstructing unsupported individual error claims.

---

# 10. Interpreting the Real-World Result

The LoRA model achieved **54.95% field accuracy on the real-world holdout**.

This result should be interpreted as a measurement of generalization under the experimental setup, rather than as a statement that LoRA itself is inherently unsuitable for invoice extraction.

Several factors can influence this result.

### Synthetic-to-real distribution shift

The fine-tuning data was primarily synthetic, while the holdout contains real-world documents.

Synthetic examples can provide clean and controlled supervision, but real receipts may contain substantially different:

* layouts
* OCR characteristics
* vendor formats
* abbreviations
* line-item structures
* tax representations
* currency formats

Consequently, a model can learn the extraction task successfully on synthetic examples while still struggling with previously unseen document distributions.

### Limited training data

The fine-tuning experiment was conducted under a constrained token budget.

A relatively small training set limits the number of document layouts and formatting patterns available during adaptation.

Increasing the diversity and coverage of training examples may therefore affect generalization.

### Variable line-item structure

Real receipts can contain substantially different numbers and arrangements of line items.

Documents with many items create a more difficult sequence-generation problem than short invoices containing only a few products.

### Context limitations

Long receipts and invoices require the model to preserve relevant information across a larger input.

If relevant information is truncated or falls outside the effective context available during inference, extraction quality can decrease.

This is a potential engineering limitation that should be investigated with the original inference outputs before being attributed to individual failures.

---

# 11. Error Taxonomy

For future evaluations, StructGen uses the following error categories.

## Document-Level Errors

```text
WRONG_VENDOR
WRONG_INVOICE_NUMBER
WRONG_DATE
WRONG_TAX
WRONG_SUBTOTAL
WRONG_TOTAL
WRONG_CURRENCY
```

## Line-Item Errors

```text
MISSING_LINE_ITEMS
EXTRA_LINE_ITEMS
WRONG_DESCRIPTION
WRONG_QUANTITY
WRONG_UNIT_PRICE
WRONG_TOTAL
```

## Generation Errors

```text
TRUNCATION
HALLUCINATION
SEMANTIC_SUBSTITUTION
```

## Annotation / Evaluation Issues

```text
ANNOTATION_AMBIGUITY
```

These categories are intentionally separated because different failure types require different engineering solutions.

For example:

```text
Wrong vendor
    → extraction / semantic understanding problem

Missing line items
    → sequence reconstruction / context problem

Invented unit price
    → hallucination / constrained extraction problem

Extra annotated item
    → potentially an annotation mismatch

Truncated output
    → context or generation-length problem
```

A future benchmark run should record these categories per example instead of relying exclusively on aggregate accuracy.

---

# 12. Synthetic vs Real-World Generalization

One of the main purposes of this benchmark is to distinguish performance on familiar synthetic data from performance on real documents.

A model can perform well on synthetic evaluation data because synthetic examples often resemble the patterns seen during training.

Real-world evaluation is more demanding because it introduces distribution shift.

Therefore:

```text
Synthetic performance
        ≠
Real-world deployment performance
```

The real-world holdout is consequently treated as the more important indicator for deployment-oriented evaluation.

---

# 13. Limitations

Several limitations should be considered when interpreting the benchmark.

### 13.1 Limited real-world evaluation set

The real-world holdout contains only a relatively small number of examples.

A small evaluation set can make aggregate metrics sensitive to individual documents.

The reported score should therefore not be interpreted as a precise estimate of performance across all invoice and receipt distributions.

### 13.2 Synthetic training data

The LoRA model was trained primarily using synthetic examples.

This introduces a potential distribution gap between training and real-world documents.

### 13.3 Limited fine-tuning budget

The experiment was performed under a constrained token budget.

This limits the amount of training data that could be used and therefore limits the diversity of examples available for adaptation.

### 13.4 Lost intermediate prediction artifacts

The original per-example result JSON files for the real-world benchmark are no longer available.

As a result, the current benchmark records the verified aggregate LoRA result but does not claim a detailed per-example error distribution.

Future benchmark runs should preserve:

```text
raw predictions
ground truth
normalized predictions
per-field metrics
line-item metrics
error classifications
```

as versioned experiment artifacts.

### 13.5 OCR / document preprocessing

The benchmark evaluates extraction from the text representation available to the models.

Errors introduced during document parsing or OCR can therefore propagate into the extraction stage.

The benchmark does not necessarily isolate:

```text
OCR errors
+
LLM extraction errors
```

into completely independent measurements.

### 13.6 Single benchmark configuration

Model performance depends on:

* model version
* prompt
* decoding parameters
* context length
* dataset composition
* preprocessing
* evaluation rules

Therefore, these results should be interpreted specifically within the configuration used for this experiment.

---

# 14. Reproducibility

Future benchmark runs should preserve the complete experiment artifact set.

Recommended structure:

```text
experiments/
├── datasets/
│   ├── train.json
│   ├── validation.json
│   └── real_holdout.json
│
├── results/
│   ├── base_qwen_real_results.json
│   ├── groq_real_results.json
│   ├── lora_real_results.json
│   └── lora_synthetic_results.json
│
├── metrics/
│   ├── document_fields.json
│   ├── line_items.json
│   └── error_analysis.json
│
└── reports/
    └── benchmark_report.md
```

Each experiment should also record:

```text
model name
model version
prompt version
dataset version
training configuration
LoRA configuration
evaluation configuration
inference parameters
timestamp
```

This prevents aggregate metrics from becoming detached from the exact experiment that produced them.

---

# 15. Engineering Takeaways

The benchmark demonstrates several important lessons about LLM-based information extraction.

### 1. Aggregate accuracy is not enough

A single score such as:

```text
54.95%
```

does not explain why a model fails.

A production-oriented evaluation system should report both aggregate metrics and structured error categories.

### 2. Fine-tuning does not automatically guarantee real-world generalization

Task-specific adaptation can improve behavior on the target task, but generalization depends heavily on the diversity and realism of the training data.

### 3. Data diversity is a critical part of fine-tuning

For invoice extraction, diversity should cover more than vendor names.

It should include:

* layouts
* currencies
* tax formats
* date formats
* receipt styles
* line-item counts
* missing fields
* noisy text
* multilingual or region-specific conventions where relevant

### 4. Structured extraction requires evaluation beyond text similarity

The goal is not simply to generate plausible text.

The model must reconstruct a valid structured representation.

Therefore, field-level correctness and line-item-level correctness are more meaningful than generic text-generation metrics.

### 5. Real-world holdouts are essential

A model that performs well on synthetic documents may still encounter substantial difficulties when exposed to real receipts.

Maintaining a completely separate real-world holdout is therefore an important part of the benchmark design.

---

# 16. Future Work

The next iteration of StructGen should focus on improving both the model and the evaluation methodology.

## Data

* Increase training-set diversity.
* Add more real-world documents.
* Include more receipt layouts.
* Increase the number of examples with long line-item tables.
* Add more currencies and regional formats.
* Include controlled noisy/OCR examples.

## Fine-Tuning

* Compare different LoRA ranks.
* Evaluate different learning rates.
* Experiment with training-set size.
* Compare 200 vs larger training subsets when the token budget permits.
* Investigate the effect of longer context windows.

## Evaluation

Implement automated:

```text
per-field accuracy
line-item precision
line-item recall
line-item F1
missing-item detection
extra-item detection
hallucination detection
truncation detection
error-category classification
```

## Benchmarking

Preserve every raw prediction artifact so that aggregate metrics can always be traced back to individual examples.

---

# 17. Current Benchmark Status

The current experiment establishes a baseline for evaluating task-specific fine-tuning against prompted general-purpose extraction.

The verified real-world LoRA result is:

> **54.95% document-field accuracy**

The result indicates that the fine-tuned model still has substantial room for improvement when moving from controlled/synthetic examples to real-world documents.

However, the experiment should not be interpreted as evidence that a particular modeling approach is universally better or worse.

The primary value of the benchmark is that it exposes the **generalization gap** and provides a framework for investigating the causes of extraction failures.

The next iteration should therefore focus on:

```text
better data diversity
        ↓
better fine-tuning coverage
        ↓
more rigorous error analysis
        ↓
real-world re-evaluation
```

rather than optimizing solely for a single aggregate accuracy number.
