import pandas as pd
from datasets import Dataset
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_CLASSES = [
    'normal',
    'sqli',
    'xss',
    'lfi',
    'rce',
    'other',
    'upload',
    'traversal'
]
CLASS_MAP = {c: i for i, c in enumerate(TARGET_CLASSES)}
NUM_CLASSES = len(TARGET_CLASSES)

def clean_label(label_str):
    if pd.isna(label_str) or not isinstance(label_str, str):
        return 'normal'
    
    cleaned = label_str.strip('[]"\' ').lower()
    
    for c in TARGET_CLASSES[1:]: # exclude normal
        if c in cleaned:
            return c
            
    return 'other' if label_str else 'normal'

def load_data(path, max_samples_per_class=5000):
    logger.info(f"Loading hybrid dataset from {path}...")
    df = pd.read_csv(path)
    
    # Balance / Subsample to speed up training
    dfs = []
    for lbl in df['label'].unique():
        sub_df = df[df['label'] == lbl]
        dfs.append(sub_df.sample(n=min(len(sub_df), max_samples_per_class), random_state=42))
    df_sampled = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"Final dataset size: {len(df_sampled)}")
    logger.info("Class distribution:")
    logger.info(df_sampled['label'].value_counts())
    
    return Dataset.from_pandas(df_sampled[['text', 'label']])

def prepare_hf_dataset(path, tokenizer, test_size=0.2, max_length=128, max_samples_per_class=5000):
    dataset = load_data(path, max_samples_per_class)
    
    # train/test split
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
        
    tokenized_datasets = split_dataset.map(tokenize_function, batched=True)
    
    return tokenized_datasets

if __name__ == "__main__":
    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    ds = prepare_hf_dataset('../../data/hybrid_dataset.csv', tokenizer, max_samples_per_class=100)
    print("Dataset prepared successfully.")
    print("Train features:", ds['train'][0])
