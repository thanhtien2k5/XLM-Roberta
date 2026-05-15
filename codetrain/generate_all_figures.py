"""
generate_all_figures.py
Tạo 9 hình ảnh cho khóa luận:
- Hình 1.1, 1.2, 2.1, 2.2, 2.3, 2.4 (vẽ schematic)
- Hình 4.1, 4.2, 4.3 (dựa trên model và dữ liệu thực tế)
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from langdetect import detect, DetectorFactory

# ----------------------------------------------
# PHẦN 1: VẼ CÁC HÌNH LÝ THUYẾT (không cần dữ liệu)
# ----------------------------------------------

def fig11_ai_content_boom(save_path="figure_1_1_ai_boom.png"):
    """Hình 1.1: Biểu đồ bùng nổ nội dung AI sinh ra"""
    years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    # Dữ liệu giả định dựa trên các báo cáo thực tế
    # (số lượng bài báo/tweet/bài viết AI)
    articles = [0.2, 0.5, 1.2, 3.5, 8.2, 18.5, 32.0]  # triệu bài
    plt.figure(figsize=(8, 5))
    plt.plot(years, articles, marker='o', linewidth=2, markersize=8,
             color='#d62728')
    plt.fill_between(years, articles, alpha=0.3, color='#d62728')
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Estimated AI-generated content (million articles/tweets)",
               fontsize=10)
    plt.title("Explosive Growth of AI-Generated Content Online", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    for x, y in zip(years, articles):
        plt.text(x, y+1, f"{y:.1f}M", ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")


def fig12_system_flow(save_path="figure_1_2_system_flow.png"):
    """Hình 1.2: Sơ đồ luồng tổng quan hệ thống phát hiện"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')
    # Các khối
    blocks = {
        "Input": (0.5, 1.2, 1.5, 0.6),
        "Preprocess": (2.5, 1.2, 1.5, 0.6),
        "XLM-RoBERTa": (4.8, 1.2, 1.8, 0.6),
        "Classifier": (7.2, 1.2, 1.3, 0.6),
        "Output": (9.0, 1.2, 0.8, 0.6)
    }
    colors = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6']
    for (name, (x, y, w, h)), col in zip(blocks.items(), colors):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              facecolor=col, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, name, ha='center', va='center',
                fontsize=10, fontweight='bold')
    # Mũi tên
    arrows = [(2.0, 1.5, 2.5, 1.5), (4.0, 1.5, 4.8, 1.5),
              (6.6, 1.5, 7.2, 1.5), (8.5, 1.5, 9.0, 1.5)]
    for (x1, y1, x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    plt.title("Overall Pipeline of Machine-Generated Text Detection",
              fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")


def fig21_transformer_arch(save_path="figure_2_1_transformer_arch.png"):
    """Hình 2.1: Kiến trúc tổng quát Transformer"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 10)
    ax.axis('off')
    # Encoder stack
    encoder_box = FancyBboxPatch((0.5, 6), 3, 3.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#cce5ff', edgecolor='black',
                                 linewidth=1.5)
    ax.add_patch(encoder_box)
    ax.text(2, 9, "Encoder", ha='center', fontsize=12, fontweight='bold')
    ax.text(2, 8.5, "x N", ha='center', fontsize=10, style='italic')
    # Decoder stack
    decoder_box = FancyBboxPatch((4.5, 6), 3, 3.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#d9f0d3', edgecolor='black',
                                 linewidth=1.5)
    ax.add_patch(decoder_box)
    ax.text(6, 9, "Decoder", ha='center', fontsize=12, fontweight='bold')
    ax.text(6, 8.5, "x N", ha='center', fontsize=10, style='italic')
    # Input Embedding
    ax.add_patch(Rectangle((1, 9.5), 1.2, 0.5, facecolor='lightgray',
                            edgecolor='black'))
    ax.text(1.6, 9.75, "Input\nEmb", ha='center', va='center', fontsize=8)
    # Output Embedding (shifted right)
    ax.add_patch(Rectangle((5.3, 9.5), 1.2, 0.5, facecolor='lightgray', edgecolor='black'))
    ax.text(5.9, 9.75, "Output\nEmb", ha='center', va='center', fontsize=8)

    # Positional Encoding
    ax.add_patch(Rectangle((2.5, 9.5), 1, 0.5, facecolor='#ffe0b3', edgecolor='black'))
    ax.text(3, 9.75, "PosEnc", ha='center', va='center', fontsize=7)
    ax.add_patch(Rectangle((4.5, 9.5), 1, 0.5, facecolor='#ffe0b3', edgecolor='black'))
    ax.text(5, 9.75, "PosEnc", ha='center', va='center', fontsize=7)

    # Mũi tên
    ax.annotate('', xy=(1.6,9.5), xytext=(1.6,9.0), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(5.9,9.5), xytext=(5.9,9.0), arrowprops=dict(arrowstyle='->'))

    # Linear + Softmax
    ax.add_patch(Rectangle((5, 0.5), 2, 0.8, facecolor='#f2b5d4', edgecolor='black'))
    ax.text(6, 0.9, "Linear + Softmax", ha='center', va='center', fontsize=8)
    ax.annotate('', xy=(6,1.3), xytext=(6,2.0), arrowprops=dict(arrowstyle='->'))

    plt.title("Transformer Architecture (Vaswani et al., 2017)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def fig22_attention_mechanism(save_path="figure_2_2_attention.png"):
    """Hình 2.2: Cơ chế Attention"""
    fig, ax = plt.subplots(figsize=(7,5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Các khối
    # Q, K, V
    ax.add_patch(Rectangle((0.5, 3.5), 1, 0.8, facecolor='#ffcc99', edgecolor='black'))
    ax.text(1, 3.9, "Q", ha='center', fontsize=12, fontweight='bold')
    ax.add_patch(Rectangle((2.5, 3.5), 1, 0.8, facecolor='#99cc99', edgecolor='black'))
    ax.text(3, 3.9, "K", ha='center', fontsize=12, fontweight='bold')
    ax.add_patch(Rectangle((4.5, 3.5), 1, 0.8, facecolor='#99ccff', edgecolor='black'))
    ax.text(5, 3.9, "V", ha='center', fontsize=12, fontweight='bold')

    # MatMul
    ax.add_patch(Rectangle((1.5, 2.2), 1, 0.6, facecolor='#f0f0f0', edgecolor='black'))
    ax.text(2, 2.5, "MatMul", ha='center', fontsize=8)
    ax.annotate('', xy=(2,2.8), xytext=(1.5,3.5), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(2,2.8), xytext=(3,3.5), arrowprops=dict(arrowstyle='->'))

    # Scale
    ax.add_patch(Rectangle((2.8, 1.2), 1, 0.6, facecolor='#f0f0f0', edgecolor='black'))
    ax.text(3.3, 1.5, "Scale", ha='center', fontsize=8)
    ax.annotate('', xy=(3.3,1.8), xytext=(2,2.2), arrowprops=dict(arrowstyle='->'))

    # Mask (opt)
    ax.add_patch(Rectangle((0, 1.2), 1, 0.6, facecolor='#ffcccc', edgecolor='black'))
    ax.text(0.5, 1.5, "Mask\n(opt)", ha='center', fontsize=7)
    ax.annotate('', xy=(2.2,1.5), xytext=(1,1.5), arrowprops=dict(arrowstyle='->', linestyle='dashed'))

    # Softmax
    ax.add_patch(Rectangle((3.8, 0.2), 1, 0.6, facecolor='#f0f0f0', edgecolor='black'))
    ax.text(4.3, 0.5, "Softmax", ha='center', fontsize=8)
    ax.annotate('', xy=(4.3,0.8), xytext=(3.3,1.2), arrowprops=dict(arrowstyle='->'))

    # MatMul cuối
    ax.add_patch(Rectangle((5, 2.2), 1, 0.6, facecolor='#f0f0f0', edgecolor='black'))
    ax.text(5.5, 2.5, "MatMul", ha='center', fontsize=8)
    ax.annotate('', xy=(5.5,2.8), xytext=(5,3.5), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(5.5,2.8), xytext=(4.3,0.8), arrowprops=dict(arrowstyle='->'))

    ax.text(0.5, 4.5, "Attention(Q,K,V) = softmax(QKᵀ/√d_k) V", fontsize=10, fontweight='bold')
    plt.title("Scaled Dot-Product Attention", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def fig23_xlmroberta_structure(save_path="figure_2_3_xlmr_structure.png"):
    """Hình 2.3: Cấu trúc XLM-RoBERTa đa ngôn ngữ"""
    fig, ax = plt.subplots(figsize=(6,8))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Các tầng
    layers = [
        ("Input (Vi / En / ...)", 8, '#d9f0f7'),
        ("Tokenization (BPE)", 7.2, '#cce5ff'),
        ("Embedding + PosEnc", 6.4, '#ffe0b3'),
        ("Transformer Encoder Block (12 layers)", 5.6, '#b2df8a'),
        ("[CLS] Token Vector (768-d)", 4.8, '#fdbf6f'),
        ("Dropout (p=0.1)", 4.0, '#f2b5d4'),
        ("Linear Classifier (768 → 2)", 3.2, '#cab2d6'),
        ("Output: Human / Machine", 2.4, '#ff9896')
    ]
    y_start = 7.5
    step = 0.9
    for i, (name, y_top, color) in enumerate(layers):
        y_bottom = y_top - 0.6
        rect = FancyBboxPatch((1.5, y_bottom), 3, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(3, y_bottom+0.3, name, ha='center', va='center', fontsize=9)
        if i < len(layers)-1:
            ax.annotate('', xy=(3, y_bottom), xytext=(3, y_bottom+0.1),
                        arrowprops=dict(arrowstyle='->', lw=1))

    plt.title("XLM-RoBERTa Architecture for Binary Classification", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def fig24_finetuning_pipeline(save_path="figure_2_4_finetuning.png"):
    """Hình 2.4: Quy trình fine-tuning XLM-RoBERTa"""
    fig, ax = plt.subplots(figsize=(10,3))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2.5)
    ax.axis('off')

    steps = [
        (0.5, 1, "Pre-trained\nXLM-RoBERTa", '#a6cee3'),
        (2.5, 1, "M4 Dataset\n(Vi+En)", '#b2df8a'),
        (4.8, 1, "Fine-tune\nwith Trainer", '#fb9a99'),
        (7.2, 1, "Evaluate\non Test Set", '#fdbf6f'),
        (9.5, 1, "Deploy / Save\nBest Model", '#cab2d6')
    ]
    for (x, y, text, col) in steps:
        rect = FancyBboxPatch((x, y), 1.6, 0.8, boxstyle="round,pad=0.05",
                              facecolor=col, edgecolor='black', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x+0.8, y+0.4, text, ha='center', va='center', fontsize=8, fontweight='bold')

    # Mũi tên
    arrow_pos = [(2.1,1.4,2.5,1.4), (4.4,1.4,4.8,1.4), (6.9,1.4,7.2,1.4), (8.8,1.4,9.5,1.4)]
    for (x1,y1,x2,y2) in arrow_pos:
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    plt.title("Fine-tuning Pipeline for Binary Classification", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

# ----------------------------------------------
# PHẦN 2: VẼ HÌNH TỪ MÔ HÌNH THỰC TẾ (4.1, 4.2, 4.3)
# ----------------------------------------------

# Cấu hình cho máy Dell G15 (RTX 3060 6GB)
MODEL_PATH = "./xlmr-m4-vi-en/best_model"   # Đường dẫn model đã train
DATA_DIR = "D:/AI/data"                     # Đường dẫn dữ liệu M4
MAX_LENGTH = 256
BATCH_SIZE = 4                              # Giảm để tránh OOM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hàm phát hiện ngôn ngữ dự phòng
DetectorFactory.seed = 42
def detect_lang(text):
    try:
        lang = detect(text)
        return 'vi' if lang == 'vi' else 'en' if lang == 'en' else 'other'
    except:
        return 'unknown'

def load_test_data(data_dir, max_samples=None):
    """Load dữ liệu test từ M4 (chỉ giữ vi/en)"""
    records = []
    jsonl_files = glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)
    print(f"Scanning {len(jsonl_files)} files...")
    for fpath in jsonl_files:
        with open(fpath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = None
                    label = None
                    if "machine_text" in obj and obj["machine_text"]:
                        text = obj["machine_text"]
                        label = 1
                    else:
                        for field in ["text","essay","abstract","document","content","body","paragraph"]:
                            if field in obj and obj[field]:
                                text = obj[field]
                                label = 0
                                break
                    if not text or len(text.strip())<10:
                        continue
                    lang = obj.get("language", None)
                    if lang is None:
                        lang = detect_lang(text)
                    if lang in ['vi','en']:
                        records.append({"text": text.strip(), "label": label, "language": lang})
                        if max_samples and len(records)>=max_samples:
                            break
                except:
                    continue
        if max_samples and len(records)>=max_samples:
            break
    print(f"Loaded {len(records)} samples.")
    return Dataset.from_list(records)

def predict(dataset, model, tokenizer, batch_size=BATCH_SIZE):
    if len(dataset)==0:
        return np.array([]), np.array([]), np.array([])
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        inputs = tokenizer(batch["text"], return_tensors="pt",
                           truncation=True, max_length=MAX_LENGTH, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(batch["label"])
        del inputs
        torch.cuda.empty_cache()
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, save_path="figure_4_1_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human','Machine'],
                yticklabels=['Human','Machine'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def plot_roc_curve(y_true, y_probs, save_path="figure_4_2_roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_probs[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1], 'k--', lw=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def plot_f1_by_language(y_true_dict, y_pred_dict, save_path="figure_4_3_f1_by_lang.png"):
    langs = [k for k in y_true_dict if len(y_true_dict[k])>0]
    if not langs:
        print("No language data.")
        return
    f1s = [f1_score(y_true_dict[lang], y_pred_dict[lang], average='binary') for lang in langs]
    plt.figure(figsize=(4,4))
    bars = plt.bar(langs, f1s, color=['#1f77b4','#ff7f0e'])
    plt.ylim(0,1)
    plt.ylabel('F1-score')
    plt.title('F1-score by Language')
    for bar, f1 in zip(bars, f1s):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{f1:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def main():
    # 1. Vẽ các hình lý thuyết
    print("=== Drawing theoretical figures ===")
    fig11_ai_content_boom()
    fig12_system_flow()
    fig21_transformer_arch()
    fig22_attention_mechanism()
    fig23_xlmroberta_structure()
    fig24_finetuning_pipeline()

    # 2. Vẽ hình từ model (cần model và data)
    print("\n=== Loading model and data for empirical figures ===")
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Skipping empirical figures.")
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE).eval()

    # Load test data (giới hạn 5000 mẫu để đảm bảo RAM/VRAM)
    test_data = load_test_data(DATA_DIR, max_samples=5000)
    if len(test_data) == 0:
        print("No test data loaded. Skipping.")
        return

    # Dự đoán toàn bộ
    y_true, y_pred, y_probs = predict(test_data, model, tokenizer)
    if len(y_true) == 0:
        return

    # Hình 4.1, 4.2
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_probs)

    # Hình 4.3: tách theo ngôn ngữ
    vi_data = test_data.filter(lambda x: x["language"]=="vi")
    en_data = test_data.filter(lambda x: x["language"]=="en")
    _, y_pred_vi, _ = predict(vi_data, model, tokenizer)
    _, y_pred_en, _ = predict(en_data, model, tokenizer)
    y_true_vi = np.array(vi_data["label"]) if len(vi_data)>0 else np.array([])
    y_true_en = np.array(en_data["label"]) if len(en_data)>0 else np.array([])
    plot_f1_by_language({'vi': y_true_vi, 'en': y_true_en},
                        {'vi': y_pred_vi, 'en': y_pred_en})

    print("\nAll figures generated successfully!")

if __name__ == "__main__":
    main()
