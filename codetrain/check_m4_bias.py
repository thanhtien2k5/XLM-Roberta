import os
import json
import glob
import random
import argparse
from collections import Counter
from pathlib import Path

def get_text_from_obj(obj):
    """Lấy text ưu tiên machine_text, nếu không thì human_text hoặc text"""
    if "machine_text" in obj and obj["machine_text"]:
        return obj["machine_text"], 1  # machine
    if "human_text" in obj and obj["human_text"]:
        return obj["human_text"], 0
    if "text" in obj and obj["text"]:
        return obj["text"], 0
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Kiểm tra bias và chất lượng data M4 đã lọc en+vi")
    parser.add_argument("--data_dir", default="D:\\AI\\data", help="Thư mục chứa các file .jsonl")
    parser.add_argument("--sample_size", type=int, default=20, help="Số mẫu mỗi lớp để in ra")
    parser.add_argument("--output_report", default="data_report.txt", help="File báo cáo đầu ra")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"❌ Thư mục {data_dir} không tồn tại. Hãy sửa --data_dir.")
        return

    # Tìm tất cả file .jsonl (hỗ trợ thư mục con)
    jsonl_files = glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)
    if not jsonl_files:
        print(f"❌ Không tìm thấy file .jsonl nào trong {data_dir}")
        return

    print(f"🔍 Tìm thấy {len(jsonl_files)} file .jsonl")
    records = []   # lưu (text, label, file_path, line_num)
    label_counts = Counter()
    text_lengths = []
    errors = []

    # Đọc tất cả
    for fpath in jsonl_files:
        rel_path = os.path.relpath(fpath, data_dir)
        with open(fpath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text, label = get_text_from_obj(obj)
                    if text is None:
                        errors.append((rel_path, line_no, "Không có text field"))
                        continue
                    records.append((text, label, rel_path, line_no))
                    label_counts[label] += 1
                    text_lengths.append(len(text))
                except json.JSONDecodeError as e:
                    errors.append((rel_path, line_no, f"JSON lỗi: {e}"))
                except Exception as e:
                    errors.append((rel_path, line_no, f"Lỗi khác: {e}"))

    total = len(records)
    if total == 0:
        print("❌ Không có record nào hợp lệ.")
        return

    # Thống kê cơ bản
    human_count = label_counts.get(0, 0)
    machine_count = label_counts.get(1, 0)
    human_ratio = human_count / total * 100
    machine_ratio = machine_count / total * 100

    print(f"\n📊 Tổng số mẫu hợp lệ: {total:,}")
    print(f"👤 Human (0): {human_count:,} ({human_ratio:.1f}%)")
    print(f"🤖 Machine (1): {machine_count:,} ({machine_ratio:.1f}%)")
    print(f"📏 Độ dài text: min={min(text_lengths)}, max={max(text_lengths)}, mean={sum(text_lengths)/len(text_lengths):.1f}")

    # Đánh giá bias sơ bộ
    print("\n🔎 ĐÁNH GIÁ BIAS:")
    if abs(human_ratio - machine_ratio) < 10:
        print("✅ Tỷ lệ Human/Machine khá cân bằng (chênh lệch <10%) -> Không bị bias do mất cân bằng dữ liệu.")
    else:
        majority = "Machine" if machine_count > human_count else "Human"
        print(f"⚠️ Tỷ lệ mất cân bằng nặng: {majority} chiếm ưu thế. Model sẽ bị bias về {majority} nếu không dùng trọng số hoặc kỹ thuật cân bằng.")

    # Lấy mẫu ngẫu nhiên để kiểm tra chất lượng
    human_samples = [(text, path, line) for (text, label, path, line) in records if label == 0]
    machine_samples = [(text, path, line) for (text, label, path, line) in records if label == 1]

    # In report ra file
    with open(args.output_report, "w", encoding="utf-8") as out:
        out.write("===== BÁO CÁO KIỂM TRA DỮ LIỆU M4 (en+vi) =====\n\n")
        out.write(f"Thư mục: {data_dir}\n")
        out.write(f"Tổng số file .jsonl: {len(jsonl_files)}\n")
        out.write(f"Tổng số mẫu hợp lệ: {total}\n")
        out.write(f"Human: {human_count} ({human_ratio:.1f}%)\n")
        out.write(f"Machine: {machine_count} ({machine_ratio:.1f}%)\n")
        out.write(f"Độ dài text: min={min(text_lengths)}, max={max(text_lengths)}, mean={sum(text_lengths)/len(text_lengths):.1f}\n\n")

        out.write("--- MẪU HUMAN (ngẫu nhiên) ---\n")
        for i, (text, path, line) in enumerate(random.sample(human_samples, min(args.sample_size, len(human_samples))), 1):
            out.write(f"\n[{i}] File: {path} (dòng {line})\n")
            out.write(f"Text: {text[:300]}{'...' if len(text)>300 else ''}\n")

        out.write("\n\n--- MẪU MACHINE (ngẫu nhiên) ---\n")
        for i, (text, path, line) in enumerate(random.sample(machine_samples, min(args.sample_size, len(machine_samples))), 1):
            out.write(f"\n[{i}] File: {path} (dòng {line})\n")
            out.write(f"Text: {text[:300]}{'...' if len(text)>300 else ''}\n")

        if errors:
            out.write("\n\n--- LỖI JSON HOẶC THIẾU TEXT ---\n")
            for err_path, line_no, msg in errors[:50]:  # giới hạn 50 lỗi
                out.write(f"{err_path} (dòng {line_no}): {msg}\n")
            if len(errors) > 50:
                out.write(f"... và {len(errors)-50} lỗi khác.\n")

    print(f"\n✅ Báo cáo chi tiết đã lưu tại: {os.path.abspath(args.output_report)}")

    # In ra một vài mẫu trực tiếp trên terminal để bạn xem nhanh
    print("\n📝 Một vài mẫu HUMAN (kiểm tra nhanh):")
    for text, _, _ in human_samples[:3]:
        print(f"  - {text[:100]}...")

    print("\n📝 Một vài mẫu MACHINE (kiểm tra nhanh):")
    for text, _, _ in machine_samples[:3]:
        print(f"  - {text[:100]}...")

if __name__ == "__main__":
    main()