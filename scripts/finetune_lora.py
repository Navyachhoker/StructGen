"""
Fine-tune Qwen2.5-1.5B-Instruct with LoRA for invoice/receipt extraction.

Final v3 training configuration used for the portfolio benchmark.

Designed to run on a Google Colab T4 GPU.
This script is NOT intended for CPU-only training.

Expected compact dataset format:
    {
        "prompt": "...",
        "completion": "..."
    }

Example:
    python scripts/finetune_lora.py \
        --data train_compact.jsonl \
        --output_dir qwen-invoice-lora-v3

Final v3 configuration:
    - Base model: Qwen/Qwen2.5-1.5B-Instruct
    - LoRA rank: 16
    - LoRA alpha: 32
    - LoRA dropout: 0.05
    - Target modules: q_proj, k_proj, v_proj, o_proj
    - Maximum sequence length: 512
    - Batch size: 1
    - Gradient accumulation: 8
    - Learning rate: 2e-4
    - Gradient checkpointing: enabled
    - FP16: enabled
    - Validation split: 90/10, seed 42
    - Save strategy: every epoch
    - Save total limit: 2
    - Save only model: enabled

The script saves a LoRA adapter, not a merged full model.
"""

import argparse
import json


from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_LENGTH = 512
VALIDATION_RATIO = 0.10
RANDOM_SEED = 42


def load_dataset(path: str) -> Dataset:
    """
    Load the compact JSONL training dataset.

    Each row contains:
        {
            "prompt": "...",
            "completion": "..."
        }
    """

    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            examples.append(
                {
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                }
            )

    return Dataset.from_list(examples)


def build_prompt(prompt: str) -> str:
    """
    Build the exact prompt format used during v3 training.

    The completion is kept separate so that the invoice text can be
    truncated without removing the target JSON.
    """

    return f"""### Instruction:
{prompt}

### Response:
"""


def tokenize_and_mask(example, tokenizer):
    """
    Tokenize one training example while preserving the completion.

    The important v3 change is that we do NOT simply truncate the entire
    prompt + completion to MAX_LENGTH.

    Instead:
        1. Tokenize the prompt and completion separately.
        2. Reserve space for the completion.
        3. If necessary, truncate the invoice/prompt portion.
        4. Keep the complete completion whenever possible.
        5. Mask prompt tokens with -100 so loss is calculated only on
           the target JSON completion.
    """

    prompt_text = build_prompt(example["prompt"])
    completion_text = example["completion"]

    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    completion_tokens = tokenizer(
        completion_text,
        add_special_tokens=False,
    )["input_ids"]

    # Reserve room for the completion.
    max_prompt_tokens = MAX_LENGTH - len(completion_tokens)

    if max_prompt_tokens < 0:
        # This should not normally happen with the compact dataset.
        # If a completion itself is too long, truncate it as a last resort.
        completion_tokens = completion_tokens[:MAX_LENGTH]
        prompt_tokens = []
    else:
        # Truncate ONLY the prompt/invoice portion.
        prompt_tokens = prompt_tokens[:max_prompt_tokens]

    input_ids = prompt_tokens + completion_tokens

    # Attention mask for all real tokens.
    attention_mask = [1] * len(input_ids)

    # Ignore prompt tokens when calculating loss.
    labels = (
        [-100] * len(prompt_tokens)
        + completion_tokens.copy()
    )

    # Pad to MAX_LENGTH.
    padding_length = MAX_LENGTH - len(input_ids)

    if padding_length > 0:
        pad_token_id = tokenizer.pad_token_id

        input_ids += [pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        labels += [-100] * padding_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to compact training JSONL.",
    )

    parser.add_argument(
        "--output_dir",
        default="models/qwen-invoice-lora-v3",
        help="Directory where the LoRA adapter will be saved.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    print(f"Loading tokenizer: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            {"pad_token": "<|pad|>"}
        )

    # ---------------------------------------------------------
    # Base model
    # ---------------------------------------------------------

    print(f"Loading base model: {BASE_MODEL}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
    )

    # Account for the added pad token if necessary.
    model.resize_token_embeddings(len(tokenizer))

    # ---------------------------------------------------------
    # LoRA
    # ---------------------------------------------------------

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    # Gradient checkpointing reduces GPU memory usage.
    model.gradient_checkpointing_enable()

    # Required when using gradient checkpointing with some
    # parameter-efficient fine-tuning configurations.
    model.enable_input_require_grads()

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    print(f"Loading dataset: {args.data}")

    dataset = load_dataset(args.data)

    print(f"Total examples: {len(dataset)}")

    # 90/10 train-validation split.
    split = dataset.train_test_split(
        test_size=VALIDATION_RATIO,
        seed=RANDOM_SEED,
    )

    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")

    # ---------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------

    tokenized_train = train_dataset.map(
        lambda example: tokenize_and_mask(
            example,
            tokenizer,
        ),
        remove_columns=train_dataset.column_names,
    )

    tokenized_eval = eval_dataset.map(
        lambda example: tokenize_and_mask(
            example,
            tokenizer,
        ),
        remove_columns=eval_dataset.column_names,
    )

    # ---------------------------------------------------------
    # Training arguments
    # ---------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.epochs,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=8,

        learning_rate=2e-4,

        fp16=True,

        logging_steps=10,

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2,

        save_only_model=True,

        report_to="none",

        seed=RANDOM_SEED,

        remove_unused_columns=False,
    )

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    print("Starting LoRA fine-tuning...")

    trainer.train()

    # ---------------------------------------------------------
    # Save adapter
    # ---------------------------------------------------------

    print(f"Saving LoRA adapter to: {args.output_dir}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("LoRA adapter saved successfully.")


if __name__ == "__main__":
    main()