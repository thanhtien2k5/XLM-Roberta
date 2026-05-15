import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import os

MODEL_PATH = "D:/AI/xlmr-m4-vi-en/best_model"

print(f"🔄 Đang tải mô hình từ: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)

print("✅ Tải mô hình thành công!")

label_map = {
    "LABEL_0": "🟢 Human - Văn bản do người viết",
    "LABEL_1": "🔴 Machine - Văn bản do máy sinh"
}

def analyze_text(text):
    if len(text.strip()) < 10:
        return "⚠️ Văn bản quá ngắn!", 0.0, ""
    
    results = classifier(text, top_k=None)
    print("🔍 Debug:", results)   # ← Xem console để debug
    
    top = results[0]
    label = label_map.get(top['label'], top['label'])
    confidence = round(top['score'] * 100, 2)
    
    return label, confidence, f"Raw score: {top['score']:.4f}"

# Giao diện (giữ nguyên như cũ)
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 Nhận Diện Văn Bản Do Máy Sinh")
    gr.Markdown("**XLM-RoBERTa-base**")

    text_input = gr.Textbox(lines=10, placeholder="Nhập văn bản...", label="Nhập văn bản")
    btn = gr.Button("Phân tích", variant="primary")
    
    label_out = gr.Label(label="Kết quả")
    conf_out = gr.Number(label="Độ tin cậy (%)")
    debug_out = gr.Textbox(label="Debug Info")

    btn.click(analyze_text, inputs=text_input, outputs=[label_out, conf_out, debug_out])

demo.launch(share=True, debug=True)