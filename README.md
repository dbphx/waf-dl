# WAF Deep Learning Engine

This project implements a Deep Learning-based Web Application Firewall (WAF) engine capable of detecting malicious HTTP payloads using advanced NLP models.

## Project Structure

- `test/`: Inference server and attack simulation scripts.
  - `server.py`: Flask-based inference server using DistilBERT.
  - `attacker.py`: Python script to simulate attacks against the server.
- `src/distilbert/`: Transformer-based model training scripts (DistilBERT).
- `src/bilstm/`: Bi-LSTM character-level model scripts (Lightweight alternative).
- `models/`: Trained model weights and exported ONNX files.
- `application/go/`: High-performance Go-based inference engine.
- `data/`: Raw text datasets and CSV files.
- `requirements.txt`: Python dependencies.

## Models

### 1. DistilBERT (Transformer)
- **Architecture**: Attention-based Transformer.
- **Strengths**: Understanding complex semantic context.
- **Location**: `src/distilbert/`

### 2. Bi-LSTM (RNN)
- **Architecture**: Bidirectional Long Short-Term Memory.
- **Strengths**: Character-level pattern matching, extremely lightweight (~5MB).
- **Location**: `src/bilstm/`

## Prerequisites

1.  **Python Environment**:
    Create a virtual environment at the project root and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Go Environment** (for Go application):
    Ensure Go is installed (version 1.21+ recommended).

## Running the Pipeline

### 1. Start the Inference Server (Python)
This starts the Flask API serving the DistilBERT model.
```bash
# Activate venv first
source venv/bin/activate
python test/server.py
```
*Server runs on `http://127.0.0.1:10001`*

### 2. Simulate Attacks (Python)
Run the attack simulation script to test the server against various payloads (SQLi, XSS, RCE, etc.).
```bash
python test/attacker.py
```

### 3. Run Bi-LSTM Prediction
To test the lightweight Bi-LSTM model:
```bash
python src/bilstm/predict_lstm.py
```

### 4. Go Integration (Production)
The Go application loads the exported ONNX model for high-performance inference.

**Run the Go Attacker** (Simulates traffic against the Python server by default):
```bash
cd application/go
go run attacker.go
```

**Run the Go WAF Server** (Standalone):
```bash
cd application/go
go run main.go
```

## Training

To retrain the models:

**DistilBERT:**
```bash
cd src/distilbert
python train.py
```

**Bi-LSTM:**
```bash
cd src/bilstm
python train_lstm.py
```
