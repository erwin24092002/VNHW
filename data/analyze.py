import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import fire


def collect_labels(data_dir):
    samples = []
    for sample_id in os.listdir(data_dir):
        sample_path = os.path.join(data_dir, sample_id)
        if not os.path.isdir(sample_path):
            continue

        json_path = os.path.join(sample_path, "label.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        for img_name, text in labels.items():
            img_path = os.path.join(sample_path, img_name)
            if not os.path.exists(img_path):
                continue
            samples.append((img_path, text))
    return samples


def analyze_dataset(root_dir, output_dir="analysis"):
    os.makedirs(output_dir, exist_ok=True)

    widths, heights, ratios = [], [], []
    text_lengths = []
    total_samples = 0
    max_len = 0
    max_text = ""

    # === Quét qua các tập train/test ===
    for split in ["train_data", "test_data"]:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            continue

        print(f"Scanning {split_dir} ...")
        samples = collect_labels(split_dir)
        for img_path, text in samples:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            ratios.append(w / h if h != 0 else 0)
            l = len(text)
            text_lengths.append(l)
            total_samples += 1
            if l > max_len:
                max_len = l
                max_text = text

    widths = np.array(widths)
    heights = np.array(heights)
    ratios = np.array(ratios)
    text_lengths = np.array(text_lengths)

    # === Thống kê cơ bản ===
    print("\n===== DATA SUMMARY =====")
    print(f"Tổng số mẫu: {total_samples}")
    print(f"Max text length: {max_len}")
    print(f"Chuỗi dài nhất: {max_text}\n")
    print(f"Width:  mean={np.mean(widths):.1f}, std={np.std(widths):.1f}, min={np.min(widths)}, max={np.max(widths)}")
    print(f"Height: mean={np.mean(heights):.1f}, std={np.std(heights):.1f}, min={np.min(heights)}, max={np.max(heights)}")
    print(f"Aspect Ratio (w/h): mean={np.mean(ratios):.2f}, std={np.std(ratios):.2f}, min={np.min(ratios):.2f}, max={np.max(ratios):.2f}")

    # === Biểu đồ 1: Scatter w-h ===
    print(widths)
    print(heights)
    exit()
    plt.figure(figsize=(8, 6))
    plt.scatter(widths, heights, s=8, alpha=0.5)
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title("Phân phối kích thước ảnh (Width vs Height)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_wh.png"), dpi=200)
    plt.close()

    # === Biểu đồ 2: Histogram aspect ratio + phân vị ===
    plt.figure(figsize=(8, 5))
    n, bins, _ = plt.hist(ratios, bins=50, alpha=0.75, color="skyblue", edgecolor="black")
    plt.xlabel("Aspect Ratio (w/h)")
    plt.ylabel("Số lượng mẫu")
    plt.title("Phân phối tỷ lệ khung hình (w/h)")
    plt.grid(True, alpha=0.3)

    # Vẽ các đường phân vị
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        val = np.percentile(ratios, p)
        plt.axvline(val, color="red", linestyle="--", linewidth=1)
        plt.text(val, max(n) * 0.9, f"{p}%", rotation=90, va="top", ha="center", color="red", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hist_ratio_percentile.png"), dpi=200)
    plt.close()

    # === Biểu đồ 3: Histogram text length ===
    plt.figure(figsize=(8, 5))
    plt.hist(text_lengths, bins=50, alpha=0.75, color="lightgreen", edgecolor="black")
    plt.xlabel("Độ dài chuỗi (số ký tự)")
    plt.ylabel("Số lượng mẫu")
    plt.title("Phân phối độ dài text")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hist_text_length.png"), dpi=200)
    plt.close()

    print(f"\nSaved plots to '{output_dir}/'")


if __name__ == "__main__":
    fire.Fire(analyze_dataset)
