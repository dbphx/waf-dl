import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import logging
import os
import argparse

from dataset import prepare_hf_dataset, TARGET_CLASSES, NUM_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train(normal_path, attack_path, output_dir, max_samples=5000, epochs=3, batch_size=16):
    logger.info("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    logger.info("Preparing dataset...")
    datasets = prepare_hf_dataset(
        path='../../data/augmented_dataset.csv', 
        tokenizer=tokenizer, 
        test_size=0.2, 
        max_length=128, 
        max_samples_per_class=max_samples
    )
    
    logger.info("Loading model...")
    # Map for config
    id2label = {i: c for i, c in enumerate(TARGET_CLASSES)}
    label2id = {c: i for i, c in enumerate(TARGET_CLASSES)}
    
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id
    )
    
    # We will use MPS (Metal Performance Shaders) or CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    logger.info(f"Using device: {device}")
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir=f"./{output_dir}",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir='./logs',
        logging_steps=50
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Evaluating...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save the final model model
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"Model saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--normal', default='../../data/normal.csv', help='Path to normal.csv')
    parser.add_argument('--attack', default='../../data/attack.csv', help='Path to attack.csv')
    parser.add_argument('--output', default='saved_model', help='Output directory for the model')
    parser.add_argument('--samples', type=int, default=5000, help='Max samples per class')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    
    train(args.normal, args.attack, args.output, args.samples, args.epochs, args.batch_size)
