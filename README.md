# XLM-Roberta

M4 dataset và mã huấn luyện / đánh giá cho mô hình phát hiện văn bản do máy sinh (XLM-RoBERTa).

## Cấu trúc

- `data/` — file `.jsonl` dataset M4
- `codetrain/` — script train, evaluate, predict, Gradio app

## Dữ liệu

Tên file human: `<DOMAIN>_human.jsonl`  
Tên file machine: `<DOMAIN>_<MODEL>.jsonl`

**DOMAIN:** `wikipedia`, `wikihow`, `reddit`, `arxiv`, …  
**MODEL:** `davinci`, `chatGPT`, `cohere`, `dolly-v2`, `bloomz`, `flan-t5`, `llama`, …

Mỗi dòng JSONL: `prompt`, `human_text`, `machine_text`, `model`, `source`, `source_ID`

## Cài đặt

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Huấn luyện

```bash
cd codetrain
python train_m4_xlmr.py --data_dir ../data
```

## Đẩy lên GitHub

```bash
git add data codetrain requirements.txt .gitignore README.md
git commit -m "Add M4 dataset and training code"
git push origin main
```
