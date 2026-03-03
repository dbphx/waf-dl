import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import argparse
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataset import TARGET_CLASSES

def predict(model_path, inputs):
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
    
    for text in inputs:
        encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = model(**encoding)
            
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()
        
        # In case we only get 1 output if inputs length was 1, make sure it's a list
        if not isinstance(probs, list):
            probs = [probs]
            
        prediction = torch.argmax(logits, dim=-1).item()
        predicted_class = TARGET_CLASSES[prediction]
        
        print(f"\n[Input]: {text}")
        print(f"[Prediction]: {predicted_class.upper()}")
        print("[Confidences]:")
        # Ensure we zip correctly
        if hasattr(probs, '__iter__'):
             for cls, prob in zip(TARGET_CLASSES, probs):
                 print(f"  - {cls}: {prob:.4f}")
        else:
             print("  Warning: single scalar probability instead of array")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='saved_model/final_model', help='Path to trained model')
    parser.add_argument('--text', nargs='+', default=[
        "GET /index.html ",
        "GET /login.php username=admin' OR '1'='1",
        "POST /comment.php <script>alert(1)</script>",
        "GET /images/../../../etc/passwd "
    ], help='Text to predict')
    args = parser.parse_args()
    
    predict(args.model, args.text)
