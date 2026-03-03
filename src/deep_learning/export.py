import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_to_onnx(model_path, output_path):
    logger.info(f"Loading PyTorch model from {model_path}...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Create dummy inputs for tracing
    # max_length is typical for the dataset
    text = "GET /index.html"
    inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Ensure export directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        }
    )
    logger.info("ONNX export completed successfully.")
    
    # Also save the tokenizer components to the same dir for Go tokenizer integration
    tokenizer_dir = os.path.join(os.path.dirname(output_path), 'tokenizer')
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    logger.info(f"Tokenizer files saved to {tokenizer_dir} for downstream integrations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='saved_model/final_model', help='Path to PyTorch model')
    parser.add_argument('--output', default='../../models/deep_learning/distilbert_waf.onnx', help='Path to output ONNX file')
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.output)
