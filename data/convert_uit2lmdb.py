import os
import json
import cv2
import lmdb
import numpy as np
import fire


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape
    return h * w > 0


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def collect_samples(data_dir):
    """
    Duyệt toàn bộ thư mục data_dir, đọc các file label.json
    Trả về list [(image_path, label_text), ...]
    """
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


def createDataset(root_dir, outputPath, split="train_data", checkValid=True):
    """
    Convert dataset thành LMDB.
    ARGS:
        root_dir   : thư mục chứa train_data/, test_data/
        outputPath : thư mục lưu LMDB
        split      : 'train_data' hoặc 'test_data'
    """
    input_dir = os.path.join(root_dir, split)
    os.makedirs(outputPath, exist_ok=True)

    print(f"Scanning {input_dir} ...")
    samples = collect_samples(input_dir)
    print(f"Found {len(samples)} samples in {split}")

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    for img_path, label in samples:
        with open(img_path, "rb") as f:
            imageBin = f.read()

        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print(f"{img_path} is not valid, skipped.")
                    continue
            except Exception as e:
                print(f"Error validating {img_path}: {e}")
                continue

        imageKey = f"image-{cnt:09d}".encode()
        labelKey = f"label-{cnt:09d}".encode()
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode("utf-8")

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print(f"Written {cnt} / {len(samples)}")

        cnt += 1

    cache["num-samples".encode()] = str(cnt - 1).encode()
    writeCache(env, cache)


if __name__ == "__main__":
    fire.Fire(createDataset)
