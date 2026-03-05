import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from flask import Flask, request, jsonify
import os
import sys

# Add src to system path to import modules from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Assume the dataset.py is in the same directory for TARGET_CLASSES
from distilbert.dataset import TARGET_CLASSES

app = Flask(__name__)

# Global model and tokenizer
model = None
tokenizer = None
device = None

def load_model():
    global model, tokenizer, device
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '../models/distilbert')
    
    print(f"Loading WAF model from {model_path}...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    print("Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    # Gather incoming payload data from form or json
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()
        
    # Combine keys and values to simulate an HTTP request payload line
    payload = " ".join([f"{k}={v}" for k, v in data.items()]).strip()
    
    if not payload:
        return jsonify({"status": "ERROR", "message": "No payload provided"}), 400

    # Tokenize and predict
    encoding = tokenizer(payload, return_tensors='pt', truncation=True, max_length=128, padding=True)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding)
        
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    prediction = torch.argmax(logits, dim=-1).item()
    predicted_class = TARGET_CLASSES[prediction]
    
    status = "NORMAL" if predicted_class.lower() == "normal" else "ATTACK"
    
    return jsonify({
        "status": status,
        "type": predicted_class,
        "payload": payload
    })

if __name__ == '__main__':
    load_model()
    app.run(host='127.0.0.1', port=10001)
