# WAF Deep Learning Engine

This directory contains the Python-based Deep Learning infrastructure for the WAF, utilizing PyTorch and Hugging Face to train multiple detection models.

## Project Structure

- `src/distilbert/`: Transformer-based model (Old Model).
- `src/bilstm/`: Bi-LSTM character-level model (New Model).
- `models/`: Trained model weights and exported ONNX files.
- `application/go/`: Go-based production inference engine.
- `data/`: Raw text datasets and generated CSVs.

## Model Comparison

We have introduced a second Deep Learning architecture to compare against the Transformer-based approach.

### 1. Model: DistilBERT (Transformers)
- **Algorithm**: Attention-based Transformer.
- **Strengths**: Pre-trained on massive text; excellent at understanding complex context.
- **Location**: `src/distilbert/train.py`

### 2. Model: Bi-LSTM (Recurrent Neural Network)
- **Algorithm**: Bidirectional Long Short-Term Memory.
- **Strengths**: Processes sequences character-by-character; tiny footprint and very fast.
- **Location**: `src/bilstm/train_lstm.py`

### Comparison Summary
| Feature | DistilBERT | Bi-LSTM |
|---------|------------|---------|
| Architecture | Attention / Transformer | Recurrent (LSTM) |
| Training Speed | Slower (Heavyweight) | Faster (Lightweight) |
| Model Size | ~260MB | ~5MB |
| Best For | Natural language context | Obfuscated pattern matching |

## Prerequisites

Navigate to either model directory in `src/` and create a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn transformers datasets torch onnx onnxscript accelerate flask requests
```

## Running the Pipeline

**1. DistilBERT Inference Server**:
```bash
cd src/distilbert
python server.py
```

**2. Bi-LSTM Inference Test**:
```bash
cd src/bilstm
python predict_lstm.py
```

**3. Attack Simulation**:
Test the server's detection capabilities using the provided simulation scripts.

**Go Attacker**:
```bash
cd application/go
go run attacker.go
```

**4. Export to ONNX for Go inference**:
```bash
cd src/distilbert
python export.py
```

## Go Integration

The exported ONNX models can be loaded by Go applications for high-performance production inference.

### Running the Go Server
The prototype in `application/go` acts as an HTTP endpoint.
```bash
cd application/go
go run main.go
```
