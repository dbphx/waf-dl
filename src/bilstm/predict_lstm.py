import torch
import torch.nn as nn
import os

class CharTokenizer:
    def __init__(self, max_len=128):
        self.max_len = max_len
        self.vocab = {chr(i): i + 1 for i in range(128)}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = len(self.vocab)
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        if not isinstance(text, str): text = str(text)
        tokens = [self.vocab.get(c, self.vocab['<UNK>']) for c in text[:self.max_len]]
        tokens += [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

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

TARGET_CLASSES = ['normal', 'sqli', 'xss', 'lfi', 'rce', 'other', 'upload', 'traversal']

def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model_path = "../../models/bilstm/waf_bilstm.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train_lstm.py first.")
        return

    tokenizer = CharTokenizer(max_len=128)
    # Match hidden_dim from train_lstm.py (changed to 64)
    model = BiLSTMClassifier(tokenizer.vocab_size, 64, 64, len(TARGET_CLASSES)).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_payloads = [
        "GET /index.html",
        "admin' OR '1'='1",
        "<script>alert(1)</script>",
        "../../../../etc/passwd",
        "eval(base64_decode('...'))"
    ]

    print(f"Testing Bi-LSTM Model ({model_path})...\n")
    print(f"{'Payload':<30} | {'Prediction':<10} | {'Confidence'}")
    print("-" * 60)

    with torch.no_grad():
        for payload in test_payloads:
            input_tensor = tokenizer.encode(payload).to(device)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, predicted = torch.max(probs, 1)
            pred_label = TARGET_CLASSES[predicted.item()]
            print(f"{payload:<30} | {pred_label.upper():<10} | {conf.item():.4f}")

if __name__ == "__main__":
    run_test()
