# VIETNAMESE HANDRITTEN TEXT RECOGNITION

The goal of this project is to build and evaluate high-performing models for recognizing Vietnamese handwritten words.  
We explore Transformer-based recognition models (ViT-Parseq and SVTRv2-Parseq) and benchmark their performance on the UIT-HWDB dataset.

---

## 1. Project Structure
```
├── configs                     # YAML configuration files for training/inference (model, dataset, optimizer, etc.)
│   ├── vit_parseq.yml          # Config file for ViT-Parseq model
│   └── svtrv2_parseq.yml       # Config file for SVTRv2-Parseq model
│
├── data                        # Dataset-related files
│   ├── character_dict          # Stores generated character dictionary (.txt)
│   ├── data_lmdb               # Processed dataset in LMDB format for efficient IO
│   └── data_raw                # Original raw data
│       ├── UIT_HWDB_line       # UIT-HWDB dataset (Vietnamese handwriting)
│       │   ├── train_data      # Training images and labels
│       │   └── test_data       # Testing images and labels
│
├── env                         # Environment setup files
│   └── environment.yml         # Conda environment specification
│
├── lib                         # Source code and external libraries
│   └── OpenOCR                 # Core OCR framework (training, evaluation, inference tools)
│
├── outputs                     # Model checkpoints, logs, and training artifacts
│   ├── svtrv2_parseq           # Training outputs for SVTRv2-Parseq
│   └── vit_parseq              # Training outputs for ViT-Parseq
│
├── rec_results                 # Inference results and recognized text output
│
└── scripts                     # Utility scripts for data training, evaluation, inference
```

---

## 2. Environment Setup
```bash
conda env create -f env/environment.yml
conda activate env_vnhw
```

---

## 3. Data Preparation

### 3.1. Step 1: Download UIT-HWDB Dataset
```bash
cd data/data_raw
gdown https://drive.google.com/uc?id=1frQAnh_AViNrNkyRXbbt2DhyGQT-k2dK
unzip UIT_HWDB_word.zip 
```

### 3.2. Step 2: Convert Dataset to LMDB Format
```bash
python convert_uit2lmdb.py --root_dir data_raw/UIT_HWDB_word --outputPath data_lmdb/train --split train_data
python convert_uit2lmdb.py --root_dir data_raw/UIT_HWDB_word --outputPath data_lmdb/test --split test_data
```

### 3.3. Step 3: Create Character Dictionary
```bash
python create_character_dict.py --root_dir data_raw/UIT_HWDB_word --output_path uit_hwdb_word.txt
```

### 3.4. Step 4: Analyze Dataset Statistics
```bash
python analyze.py --root_dir data_raw/UIT_HWDB_word
```
```
Sample Output:
===== DATA SUMMARY =====
Tổng số mẫu: 110488
Max text length: 11
Chuỗi dài nhất: Environment
Width:  mean=128.0, std=0.0, min=128, max=128
Height: mean=98.3, std=32.3, min=23, max=974
Aspect Ratio (w/h): mean=1.43, std=0.43, min=0.13, max=5.57
```

---

## 4. Training

### 4.1. ViT-Parseq Model
```bash
CUDA_VISIBLE_DEVICES=1 python lib/OpenOCR/tools/train_rec.py --c ./configs/vit_parseq.yml
```

### 4.2. SVTRv2-Parseq Model
```bash
CUDA_VISIBLE_DEVICES=1 python lib/OpenOCR/tools/train_rec.py --c ./configs/svrtv2_parseq.yml
```

---

## 5. Evaluation

### 5.1. ViT-Parseq
```bash
CUDA_VISIBLE_DEVICES=1 python lib/OpenOCR/tools/eval_rec.py --c ./configs/vit_parseq.yml
```

### 5.2. SVTRv2-Parseq
```bash
CUDA_VISIBLE_DEVICES=1 python lib/OpenOCR/tools/eval_rec.py --c ./configs/svrtv2_parseq.yml
```
```
Sample Output:
[2025/10/10 06:11:33] openrec INFO: metric eval ***************
[2025/10/10 06:11:33] openrec INFO: acc:0.9142658767293791
[2025/10/10 06:11:33] openrec INFO: norm_edit_dis:0.9618072429880578
[2025/10/10 06:11:33] openrec INFO: num_samples:2881
[2025/10/10 06:11:33] openrec INFO: fps:1133.429803231513
```
---

## 6. Inference

Edit the config file to specify the test image directory, e.g.: ```infer_img: data/data_raw/UIT_HWDB_word/test_data/250```

### 6.1. ViT-Parseq
```bash
CUDA_VISIBLE_DEVICES=1 python lib/OpenOCR/tools/infer_rec.py --c ./configs/vit_parseq.yml
```
### 6.2. SVTRv2-Parseq
```bash
CUDA_VISIBLE_DEVICES=1 python lib/OpenOCR/tools/infer_rec.py --c ./configs/svrtv2_parseq.yml
```
```
Sample Output:
rec_results/rec_result.txt
data/data_raw/UIT_HWDB_word/test_data/250/1.jpg    Thứ    0.9974513649940491
data/data_raw/UIT_HWDB_word/test_data/250/10.jpg   Võ     0.7862417697906494
...
data/data_raw/UIT_HWDB_word/test_data/250/101.jpg  ít     0.9992736577987671
```
---

## 7. References

[OpenOCR: A general OCR system with accuracy and efficiency](https://github.com/Topdu/OpenOCR)

[Scene Text Recognition with Permuted Autoregressive Sequence Models](https://arxiv.org/pdf/2207.06966)

[Svtrv2: CTC beats encoder-decoder models in scene text recognition (arXiv 2024)](https://arxiv.org/abs/2411.15858)

[UIT-HWDB: Using Transferring Method to Construct a Novel Benchmark for Evaluating Unconstrained Handwriting Image Recognition in Vietnamese (IEEE RIVF 2022)](https://doi.org/10.1109/RIVF55975.2022.10013898)

---

## Acknowledgment

This work was developed as part of a university course project on Handwritten Text Recognition.  
It integrates modern OCR architectures (Parseq & SVTRv2) with the Vietnamese handwritten dataset UIT-HWDB,  
aiming to benchmark recognition accuracy and efficiency for the Vietnamese language.
