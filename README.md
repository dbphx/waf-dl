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
- `server.py`: A Flask-based HTTP server that loads the trained model for real-time inference.
- `attacker.py`: A Python script to simulate adversarial HTTP traffic against the inference server.
- `test_from_file.py`: A testing script designed to read payloads line-by-line from `data/attack.txt` or `data/normal.txt` to calculate sample-set accuracy.
- `export.py`: The bridge to the Go runtime. It converts the saved PyTorch model to `ONNX` format, saving it to `/models/deep_learning/distilbert_waf.onnx`.

## Model Comparison (`model_train` folder)

We have introduced a second Deep Learning architecture to compare against the Transformer-based approach.

### 1. Old Model: DistilBERT (Transformers)
- **Algorithm**: Attention-based Transformer.
- **Strengths**: Pre-trained on massive text datasets; excellent at understanding complex language context even with small fine-tuning data.
- **Location**: `src/deep_learning/train.py`

### 2. New Model: Bi-LSTM (Recurrent Neural Network)
- **Algorithm**: Bidirectional Long Short-Term Memory.
- **Strengths**: Processes sequences character-by-character; lightweight and efficient for detecting sequential patterns in obfuscated payloads.
- **Location**: `model_train/train_lstm.py`

### Comparison Summary
| Feature | DistilBERT | Bi-LSTM |
|---------|------------|---------|
| Architecture | Attention / Transformer | Recurrent (LSTM) |
| Training Speed | Slower (Heavyweight) | Faster (Lightweight) |
| Cold Start Performance | Excellent (Pre-trained) | Poor (Requires more data) |
| Model Size | ~260MB | ~5MB |

## Prerequisites

Navigate to `src/deep_learning` and create a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn transformers datasets torch onnx onnxscript accelerate flask requests
```

## Running the Pipeline

**1. Train the model**:
By default, the script trains over a limited balance of datasets using max samples. For the full dataset, increase `--samples`.
```bash
python train.py --samples 50000 --epochs 3 --batch-size 32
```

**2. Real-time Inference Server**:
Start the Flask server to receive HTTP requests for classification.
```bash
python server.py
```

**3. Attack Simulation**:
Test the server's detection capabilities using the provided simulation scripts (available in Python and Go).

**Python Attacker**:
```bash
python attacker.py
```

**Go Attacker**:
```bash
cd ../../application/go
go run attacker.go
```

**4. Export to ONNX for Go inference**:
```bash
python export.py
```

## Go Integration

The resulting ONNX model (`distilbert_waf.onnx`) can be loaded by Go applications for high-performance production inference.

### 1. Running the Go Server
The prototype in `application/go` has been updated to act as an HTTP endpoint.
```bash
cd application/go
go run main.go
```

### 2. Testing with Go Attacker
Run the Go-based attacker to verify the model's performance against various encodings (URL, Hex, Base64).
```bash
cd application/go
go run attacker.go
```
