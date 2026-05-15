"""
evaluate_and_plot.py (fixed)
=============================
Vẽ các đồ thị cho khóa luận, xử lý trường hợp thiếu trường 'language'.
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, f1_score,
    accuracy_score, precision_score, recall_score
)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from langdetect import detect, DetectorFactory

# Fix seed cho langdetect
DetectorFactory.seed = 42

# ------------------------------
# CẤU HÌNH
# ------------------------------
MODEL_PATH = "./xlmr-m4-vi-en/best_model"
DATA_DIR = "D:/AI/data"          # thư mục gốc của M4
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# 1. Load model và tokenizer
# ------------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE).eval()

# ------------------------------
# 2. Hàm phát hiện ngôn ngữ dự phòng
# ------------------------------
def detect_language(text):
    try:
        lang = detect(text)
        return 'vi' if lang == 'vi' else 'en' if lang == 'en' else 'other'
    except:
        return 'unknown'

# ------------------------------
# 3. Load dữ liệu test từ M4 (chỉ lấy các mẫu có nhãn rõ ràng)
# ------------------------------
def load_test_data(data_dir, use_langdetect_fallback=True):
    """Load tất cả file .jsonl, chỉ giữ lại mẫu có text và label hợp lệ."""
    all_records = []
    jsonl_files = glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)
    
    print(f"Scanning {len(jsonl_files)} JSONL files...")
    for fpath in jsonl_files:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    
                    # --- Lấy text ---
                    text = None
                    label = None
                    if "machine_text" in obj and obj["machine_text"]:
                        text = obj["machine_text"]
                        label = 1
                    else:
                        for field in ["text", "essay", "abstract", "document", "content", "body", "paragraph"]:
                            if field in obj and obj[field] and isinstance(obj[field], str):
                                text = obj[field]
                                label = 0
                                break
                    if not text or len(text.strip()) < 10:
                        continue
                    
                    # --- Xác định ngôn ngữ ---
                    lang = obj.get("language", None)
                    if lang is None and use_langdetect_fallback:
                        lang = detect_language(text)
                    elif lang is None:
                        lang = "unknown"
                    
                    # Chỉ giữ lại các mẫu có ngôn ngữ là vi hoặc en (hoặc cả hai nếu muốn)
                    if lang in ["vi", "en"]:
                        all_records.append({
                            "text": text.strip(),
                            "label": label,
                            "language": lang
                        })
                except Exception as e:
                    continue
    
    print(f"Loaded {len(all_records)} samples (vi/en only).")
    if len(all_records) == 0:
        print("WARNING: No samples with language='vi' or 'en' found. Check your data directory.")
        # In ra một vài file mẫu để debug
        sample_files = jsonl_files[:3]
        for sf in sample_files:
            print(f"Sample file: {sf}")
            with open(sf, encoding='utf-8') as f:
                first_line = f.readline()
                print(f"First line: {first_line[:200]}")
    return Dataset.from_list(all_records)

def split_by_language(dataset):
    """Trả về hai dataset riêng cho vi và en, kiểm tra rỗng."""
    vi_data = dataset.filter(lambda x: x["language"] == "vi")
    en_data = dataset.filter(lambda x: x["language"] == "en")
    print(f"Vietnamese samples: {len(vi_data)}, English samples: {len(en_data)}")
    return vi_data, en_data

# ------------------------------
# 4. Inference
# ------------------------------
def predict_dataset(dataset, model, tokenizer, batch_size=32):
    if len(dataset) == 0:
        return np.array([]), np.array([]), np.array([]), []
    
    model.eval()
    all_preds, all_probs, all_labels, all_langs = [], [], [], []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        texts = batch["text"]
        labels = batch["label"]
        langs = batch.get("language", ["unknown"]*len(texts))
        
        inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels)
        all_langs.extend(langs)
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_langs

# ------------------------------
# 5. Vẽ biểu đồ
# ------------------------------
def plot_confusion_matrix(y_true, y_pred, save_path="figure_4_1_confusion_matrix.png"):
    if len(y_true) == 0:
        print("No data for confusion matrix.")
        return
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Human", "Machine"],
                yticklabels=["Human", "Machine"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix on Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved confusion matrix to {save_path}")

def plot_roc_curve(y_true, y_probs, save_path="figure_4_2_roc_curve.png"):
    if len(y_true) == 0:
        print("No data for ROC curve.")
        return
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved ROC curve to {save_path}")

def plot_f1_by_language(y_true_dict, y_pred_dict, save_path="figure_4_3_f1_by_lang.png"):
    # Loại bỏ ngôn ngữ không có dữ liệu
    available = [lang for lang in y_true_dict if len(y_true_dict[lang]) > 0]
    if not available:
        print("No language-specific data for F1 comparison.")
        return
    f1_scores = [f1_score(y_true_dict[lang], y_pred_dict[lang], average='binary') for lang in available]
    
    plt.figure(figsize=(5, 5))
    bars = plt.bar(available, f1_scores, color=['#1f77b4', '#ff7f0e'])
    plt.ylim(0, 1)
    plt.ylabel("F1-score")
    plt.title("F1-score Comparison by Language")
    for bar, f1 in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{f1:.3f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved F1-by-language chart to {save_path}")

# ------------------------------
# MAIN
# ------------------------------
def main():
    print("Loading test data...")
    test_data = load_test_data(DATA_DIR, use_langdetect_fallback=True)
    if len(test_data) == 0:
        print("ERROR: No test data loaded. Exiting.")
        return
    
    print(f"Total test samples: {len(test_data)}")
    
    # Dự đoán toàn bộ test
    print("Running inference on test set...")
    y_true, y_pred, y_probs, _ = predict_dataset(test_data, model, tokenizer)
    
    if len(y_true) == 0:
        print("No predictions generated. Exiting.")
        return
    
    # Vẽ hình 4.1 và 4.2 (luôn có dữ liệu)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_probs)
    
    # Vẽ hình 4.3: F1 theo ngôn ngữ
    vi_data, en_data = split_by_language(test_data)
    y_true_vi, y_pred_vi, _, _ = predict_dataset(vi_data, model, tokenizer)
    y_true_en, y_pred_en, _, _ = predict_dataset(en_data, model, tokenizer)
    
    y_true_dict = {'vi': y_true_vi, 'en': y_true_en}
    y_pred_dict = {'vi': y_pred_vi, 'en': y_pred_en}
    plot_f1_by_language(y_true_dict, y_pred_dict)
    
    # In các chỉ số toàn cục
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("\n===== GLOBAL METRICS =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

if __name__ == "__main__":
    main()