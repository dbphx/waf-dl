# WAF Deep Learning Engine

This directory contains the Python-based Deep Learning infrastructure for the WAF, utilizing the PyTorch and Hugging Face `transformers` libraries to train a `distilbert-base-uncased` NLP model.

## Overview

The model is trained as a sequence classifier over 8 classes:
- `normal`
- `sqli`
- `xss`
- `lfi`
- `rce`
- `other`
- `upload`
- `traversal`

The resulting output is a probability array for these classes, providing greater context into incoming web traffic payload intents rather than a simplistic 1 or 0 binary flag.

## Files

- `txt_to_df.py`: Handles parsing `data/normal.txt` and `data/attack.txt` explicitly matching string categories to 8 specific attack buckets and emitting `hybrid_dataset.csv`.
- `dataset.py`: Handles parsing the newly generated `hybrid_dataset.csv` balancing class counts to prevent generic attack dominance. It creates `Dataset` objects via the DistilBert tokenizer.
- `train.py`: The main PyTorch training loop using the `Trainer` class. Configured for MPS, CUDA, or CPU seamlessly.
- `predict.py`: A local inference script that allows verifying predictions for custom strings via `--text`.
- `test_from_file.py`: A testing script designed to read payloads line-by-line from `data/attack.txt` or `data/normal.txt` to calculate sample-set accuracy.
- `export.py`: The bridge to the Go runtime. It converts the saved PyTorch model to `ONNX` format, saving it to `/models/deep_learning/distilbert_waf.onnx`.

## Prerequisites

Navigate to `src/deep_learning` and create a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn transformers datasets torch onnx onnxscript accelerate
```

## Running the Pipeline

**1. Train the model**:
By default, the script trains over a limited balance of datasets using max samples. For the full dataset, increase `--samples`.
```bash
python train.py --samples 50000 --epochs 3 --batch-size 32
```

**2. Test predictions manually**:
```bash
python predict.py --text "GET /index.php" "GET /login?user=admin' OR '1'='1"
```

**3. Test against raw text datasets**:
```bash
python test_from_file.py --file ../../data/normal.txt
python test_from_file.py --file ../../data/attack.txt
```

**4. Export to ONNX for Go inference**:
```bash
python export.py
```

**5. Running the Go Prototype**:
The resulting ONNX model (`distilbert_waf.onnx`) is designed to be loaded by the `github.com/yalue/onnxruntime_go` library. 
Ensure you have the `libonnxruntime` dynamic libraries downloaded and placed in your system's library path (e.g., `/usr/local/lib/`).

You can review or run the example boilerplate implementation in `application/go`:

```bash
cd ../../application/go
go run main.go
```
