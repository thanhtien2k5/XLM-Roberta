import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import load_dataset, concatenate_datasets, DatasetDict
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import glob

# ====================== FOCAL LOSS & TRAINER ======================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class WeightedFocalTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.focal_loss = FocalLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        focal = self.focal_loss(logits, labels)
        if self.class_weights is not None:
            ce = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))(logits, labels)
            loss = 0.65 * focal + 0.35 * ce
        else:
            loss = focal

        return (loss, outputs) if return_outputs else loss


# ====================== LOAD DATA - SIÊU ROBUST ======================
def _normalize_text(val):
    """Convert any value (list or string) to normalized string"""
    if val is None or val == "":
        return ""
    if isinstance(val, list):
        return " ".join([str(v).strip() for v in val if v])
    return str(val).strip()

def load_and_balance_m4_data(data_dir="data", seed=42):
    print("🔄 Đang load tất cả file .jsonl...")

    all_files = glob.glob(f"{data_dir}/**/*.jsonl", recursive=True)
    print(f"Tìm thấy {len(all_files)} file .jsonl")

    datasets_list = []
    for file in all_files:
        try:
            # Load từng file với features linh hoạt
            ds = load_dataset("json", data_files=file, split="train")
            # Chỉ giữ các cột cần thiết và chuẩn hóa nhãn
            ds = ds.map(lambda x: {
                'text': _normalize_text(
                    x.get('machine_text') or x.get('human_text') or 
                    x.get('text') or x.get('prompt')
                ),
                'label': 1 if x.get('machine_text') or x.get('label') == 1 else 0
            }, remove_columns=ds.column_names)
            datasets_list.append(ds)
            print(f"✓ Loaded: {file} - {len(ds)} mẫu")
        except Exception as e:
            print(f"⚠️ Bỏ qua file: {file} - {e}")

    if not datasets_list:
        raise ValueError("Không load được file nào!")

    # Kết hợp tất cả
    dataset = concatenate_datasets(datasets_list)
    print(f"Tổng mẫu sau khi load: {len(dataset)}")

    # Tách Human và Machine
    human = dataset.filter(lambda x: x['label'] == 0)
    machine = dataset.filter(lambda x: x['label'] == 1)

    print(f"Human: {len(human)} | Machine: {len(machine)}")

    # Cân bằng (1:2)
    target_machine = len(human) * 2
    if len(machine) > target_machine:
        machine = machine.shuffle(seed=seed).select(range(target_machine))

    balanced = concatenate_datasets([human, machine]).shuffle(seed=seed)

    print(f"✅ Sau cân bằng: {len(balanced)} mẫu")
    print("Phân bố:", Counter(balanced['label']))

    # Chia dataset
    train_test = balanced.train_test_split(test_size=0.2, seed=seed)
    train_val = train_test['train'].train_test_split(test_size=0.1, seed=seed)

    return DatasetDict({
        'train': train_val['train'],
        'validation': train_val['test'],
        'test': train_test['test']
    })


# ====================== METRICS ======================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    return {
        'accuracy': round(acc, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }


# ====================== MAIN ======================
if __name__ == "__main__":
    dataset = load_and_balance_m4_data(data_dir="data")

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # Tokenize dataset
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512, padding=True)
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['text'])
    
    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
    )

    class_weights = torch.tensor([3.5, 1.0], dtype=torch.float32)

    trainer = WeightedFocalTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("🚀 Bắt đầu huấn luyện...")
    trainer.train()

    print("\n📊 Đánh giá tập test:")
    test_results = trainer.evaluate(dataset['test'])
    print(test_results)

    trainer.save_model("./best_model_final")
    tokenizer.save_pretrained("./best_model_final")
    print("✅ HOÀN TẤT! Mô hình đã lưu tại ./best_model_final")