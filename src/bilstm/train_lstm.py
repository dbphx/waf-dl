import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# Configuration
MAX_LEN = 128
EMBED_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.005

# Define Target Classes (Same as old model)
TARGET_CLASSES = ['normal', 'sqli', 'xss', 'lfi', 'rce', 'other', 'upload', 'traversal']
CLASS_MAP = {c: i for i, c in enumerate(TARGET_CLASSES)}

class CharTokenizer:
    def __init__(self, max_len=128):
        self.max_len = max_len
        # Simple ASCII-based vocabulary
        self.vocab = {chr(i): i + 1 for i in range(128)}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = len(self.vocab)
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        if not isinstance(text, str): text = str(text)
        tokens = [self.vocab.get(c, self.vocab['<UNK>']) for c in text[:self.max_len]]
        # Padding
        tokens += [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

class WAFDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return self.tokenizer.encode(text), torch.tensor(label, dtype=torch.long)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Global max pooling over the sequence dimension
        out, _ = torch.max(lstm_out, dim=1)
        out = self.fc(self.dropout(out))
        return out

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Load data
    data_path = os.path.join(script_dir, '../../data/hybrid_dataset.csv')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run txt_to_df.py first.")
        return

    df = pd.read_csv(data_path).dropna()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    tokenizer = CharTokenizer(MAX_LEN)
    train_ds = WAFDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
    test_ds = WAFDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = BiLSTMClassifier(tokenizer.vocab_size, EMBED_DIM, HIDDEN_DIM, len(TARGET_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Starting Bi-LSTM Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    present_labels = sorted(list(set(all_labels) | set(all_preds)))
    present_names = [TARGET_CLASSES[i] for i in present_labels]

    print("\n--- Bi-LSTM Classification Report ---")
    print(classification_report(all_labels, all_preds, labels=present_labels, target_names=present_names, zero_division=0))
    
    # Save model
    model_dir = os.path.join(script_dir, "../../models/bilstm")
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "waf_bilstm.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train()
