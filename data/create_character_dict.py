import os
import json
import fire


def collect_labels(data_dir):
    texts = []
    for sample_id in os.listdir(data_dir):
        sample_path = os.path.join(data_dir, sample_id)
        if not os.path.isdir(sample_path):
            continue

        json_path = os.path.join(sample_path, "label.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        texts.extend(labels.values())
    return texts


def create_character_dict(root_dir, output_path="character_dict.txt"):
    char_set = set()
    for split in ["train_data", "test_data"]:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            continue
        print(f"Scanning {split_dir} ...")
        texts = collect_labels(split_dir)
        for text in texts:
            for ch in text:
                char_set.add(ch)
    sorted_chars = sorted(list(char_set))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ch in sorted_chars:
            if ch == "\n":
                continue
            f.write(ch + "\n")
    print(f"Saved {len(sorted_chars)} unique characters to {output_path}")


if __name__ == "__main__":
    fire.Fire(create_character_dict)
