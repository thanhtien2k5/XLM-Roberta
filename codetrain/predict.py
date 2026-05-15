"""
predict.py — Dùng model đã train để phân loại văn bản
======================================================
Chạy:
    python predict.py --text "Đây là văn bản cần kiểm tra."
    python predict.py --file input.txt
    python predict.py  # Demo với 3 câu mẫu
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_MAP = {0: "✅ Human-written", 1: "🤖 Machine-generated"}
DEMO_TEXTS = [
    "mình thấy cái này cũng đúng mà không biết áp dụng kiểu gì luôn",
    "Học sinh cần phát triển tư duy phê phán để có thể đánh giá "
    "thông tin một cách chính xác trong thời đại số.",
    "Machine learning is a subset of artificial intelligence that "
    "enables systems to automatically learn and improve from "
    "experience without being explicitly programmed.",
    "This essay explores the multifaceted dimensions of climate "
    "change, examining its far-reaching implications on global "
    "ecosystems, human societies, and future generations.",
]


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, device


def predict(texts: list, tokenizer, model, device, max_length: int = 512):
    results = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

        pred = int(torch.argmax(logits, dim=-1))
        results.append({
            "label":        pred,
            "label_str":    LABEL_MAP[pred],
            "prob_human":   round(probs[0] * 100, 1),
            "prob_machine": round(probs[1] * 100, 1),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./xlmr-m4-vi-en/best_model")
    parser.add_argument("--text",       type=str, help="Văn bản cần phân loại")
    parser.add_argument("--file", type=str,
                        help="File .txt (mỗi dòng 1 văn bản)")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("(Demo mode — không có --text hay --file)\n")
        texts = DEMO_TEXTS

    print("Loading model...")
    tokenizer, model, device = load_model(args.model_path)
    print(f"Model loaded on: {device}\n")

    results = predict(texts, tokenizer, model, device, args.max_length)

    print("=" * 65)
    print("KẾT QUẢ PHÂN LOẠI")
    print("=" * 65)
    for i, (text, r) in enumerate(zip(texts, results), 1):
        preview = text[:75] + "..." if len(text) > 75 else text
        print(f"\n[{i}] {preview}")
        print(f"     → {r['label_str']}")
        print(f"        Human: {r['prob_human']}%  |  "
              f"Machine: {r['prob_machine']}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
