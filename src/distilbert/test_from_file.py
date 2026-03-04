import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import argparse
import sys
import logging
import os
from dataset import TARGET_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_from_file(model_path, file_path):
    logger.info(f"Loading model from {model_path}...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
        
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Skip the first two header lines
    lines = lines[2:]
    
    correct = 0
    total = 0
    
    expected_label = 'normal' if 'normal' in file_path.lower() else 'attack'
    
    logger.info(f"Testing samples from {file_path}")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Parse payload: typically "1. Category: payload"
        parts = line.split(': ', 1)
        if len(parts) > 1:
            text = parts[1]
        else:
            text = line
            
        encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
        # Avoid passing to device if tensors are empty or incorrectly formed.
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = model(**encoding)
            
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        predicted_class = TARGET_CLASSES[prediction]
        
        is_attack_pred = predicted_class != 'normal'
        
        # Simple accuracy check
        if expected_label == 'normal':
            if not is_attack_pred:
                correct += 1
        elif expected_label == 'attack':
            if is_attack_pred:
                correct += 1
                
        total += 1
        
        status = 'PASS' if (expected_label == 'normal' and not is_attack_pred) or (expected_label == 'attack' and is_attack_pred) else 'FAIL'
        print(f"[{status}] Payload: {text[:50]:<50} -> Pred: {predicted_class.upper()}")

    if total > 0:
        logger.info(f"Accuracy on {file_path}: {correct}/{total} ({(correct/total)*100:.2f}%)")
    else:
        logger.info("No valid test cases found.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(script_dir, '../../models/distilbert')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=default_model, help='Path to trained model')
    parser.add_argument('--file', required=True, help='Path to text file (e.g., ../../data/attack.txt)')
    args = parser.parse_args()
    
    predict_from_file(args.model, args.file)
