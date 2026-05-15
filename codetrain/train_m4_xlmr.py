"""
train_m4_xlmr.py — Train chỉ trên tiếng Việt + tiếng Anh từ dataset

"""

import os
import json
import argparse
import logging
import numpy as np
import glob
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
from torch.nn import CrossEntropyLoss


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = torch.tensor([1.0, 1.2]).to(logits.device)
        loss_fct = CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    data_dir: str = "D:\\AI\\data"
    max_length: int = 256
    model_name: str = "xlm-roberta-base"
    output_dir: str = "./xlmr-m4-vi-en"
    epochs: int = 3
    batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    fp16: bool = True
    seed: int = 42


TEXT_FIELDS = [
    "text", "essay", "abstract", "document", "content", "body", "paragraph"
]


def normalize_record(obj: dict, filename: str = ""):
    """Lấy text từ bất kỳ trường nào và chuẩn hóa label"""
    text = ""
    label = 0  # default human
    if ("machine_text" in obj and obj["machine_text"] and
            isinstance(obj["machine_text"], str)):
        text = obj["machine_text"]
        label = 1
    else:
        for field in TEXT_FIELDS:
            if field in obj and obj[field] and isinstance(obj[field], str):
                text = obj[field]
                break
    if not text:
        return None

    return {"text": text.strip(), "label": int(label)}


def load_m4_data(cfg: Config) -> DatasetDict:
    logger.info(f"Đang quét tất cả file .jsonl trong: {cfg.data_dir}")

    all_records = []
    jsonl_files = glob.glob(
        os.path.join(cfg.data_dir, "**", "*.jsonl"), recursive=True
    )

    for fpath in jsonl_files:
        filename = os.path.basename(fpath)
        count = 0
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    record = normalize_record(obj, filename)
                    if record:
                        all_records.append(record)
                        count += 1
                except Exception:
                    continue
        if count > 0:
            logger.info(
                f"  ✓ Đọc {count:>6,} records ← {filename} "
                f"({os.path.getsize(fpath)/1e6:.1f} MB)"
            )

    if len(all_records) == 0:
        raise RuntimeError(
            "Không load được record nào. Kiểm tra file .jsonl có trường "
            "'text' hoặc 'essay' không."
        )

    logger.info(f"\nTổng load được: {len(all_records):,} mẫu từ tất cả file")

    # Thống kê label
    labels = [r["label"] for r in all_records]
    n_human = labels.count(0)
    n_machine = labels.count(1)
    logger.info(
        f"Human: {n_human:,} ({n_human/len(all_records)*100:.1f}%) | "
        f"Machine: {n_machine:,} ({n_machine/len(all_records)*100:.1f}%)"
    )

    dataset = Dataset.from_list(all_records).shuffle(seed=cfg.seed)

    # Chia train/val/test
    train_val = dataset.train_test_split(test_size=0.2, seed=cfg.seed)
    val_test = train_val["test"].train_test_split(test_size=0.5, seed=cfg.seed)

    return DatasetDict({
        "train": train_val["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })


def tokenize_dataset(dataset, tokenizer, cfg):
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding=False
        )
    return dataset.map(
        tokenize_fn, batched=True, batch_size=512, desc="Tokenizing"
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1_macro": round(f1_score(labels, preds, average="macro"), 4),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="D:\\AI\\data")
    parser.add_argument("--output_dir", default="./xlmr-m4-vi-en")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--no_fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        fp16=not args.no_fp16,
    )

    set_seed(cfg.seed)

    logger.info(
        "🚀 GPU: {} | VRAM: {:.1f} GB".format(
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9
        )
    )

    logger.info("\n── BƯỚC 1: Load dữ liệu từ tất cả file .jsonl ──")
    dataset = load_m4_data(cfg)

    logger.info("\n── BƯỚC 2: Load model ──")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2
    )
    model.gradient_checkpointing_enable()

    logger.info("\n── BƯỚC 3: Tokenize ──")
    tokenized = tokenize_dataset(dataset, tokenizer, cfg)

    train_ds = tokenized["train"].select_columns(
        ["input_ids", "attention_mask", "label"]
    )
    val_ds = tokenized["validation"].select_columns(
        ["input_ids", "attention_mask", "label"]
    )

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    logger.info("\n── BƯỚC 4: Training ──")
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        fp16=cfg.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    save_path = os.path.join(cfg.output_dir, "best_model")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(
        f"\n✅ Training hoàn tất! Model lưu tại: {os.path.abspath(save_path)}"
    )


if __name__ == "__main__":
    main()
