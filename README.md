# XLM-Roberta

Phát hiện văn bản do máy sinh (Machine-Generated Text Detection) trên tập **M4** bằng **XLM-RoBERTa-base**, hỗ trợ tiếng **Việt** và **Anh**.

## Cấu trúc repo

| Thư mục / file | Mô tả |
|----------------|--------|
| [`data/`](data/) | 28 file `.jsonl` dataset M4 (~507 MB) |
| [`codetrain/`](codetrain/) | Script train, evaluate, predict, Gradio |
| [`HUONG_DAN_TRAIN.md`](HUONG_DAN_TRAIN.md) | **Hướng dẫn huấn luyện & kết quả thực nghiệm** |
| `requirements.txt` | Thư viện Python |

## Bắt đầu nhanh

```bash
git clone https://github.com/thanhtien2k5/XLM-Roberta.git
cd XLM-Roberta
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

cd codetrain
python train_m4_xlmr.py --data_dir ../data --output_dir ../xlmr-m4-vi-en
```

## Kết quả thực nghiệm (tóm tắt)

Huấn luyện `train_m4_xlmr.py`, 3 epoch, backbone `xlm-roberta-base`:

| Metric | Validation (epoch 3) |
|--------|----------------------|
| **F1 macro** | **92,28%** |
| Accuracy | 99,73% |

Chi tiết siêu tham số, biểu đồ đánh giá, xử lý lỗi và pipeline nâng cao (`train_m4_xlmr_final.py`): xem **[HUONG_DAN_TRAIN.md](HUONG_DAN_TRAIN.md)**.

## Dữ liệu

- Human: `<DOMAIN>_human.jsonl`
- Machine: `<DOMAIN>_<MODEL>.jsonl`
- Mỗi dòng: `prompt`, `human_text`, `machine_text`, `model`, `source`, `source_ID`

Xem [`data/README.md`](data/README.md).

## Script chính

| File | Chức năng |
|------|-----------|
| `codetrain/train_m4_xlmr.py` | Train cơ bản (Weighted CE) |
| `codetrain/train_m4_xlmr_final.py` | Train Focal Loss + cân bằng lớp |
| `codetrain/predict.py` | Dự đoán trên văn bản |
| `codetrain/evaluate_and_plot.py` | Đánh giá + vẽ biểu đồ |
| `codetrain/app.py` | Giao diện Gradio |
