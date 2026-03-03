# WAF Deep Learning Attack Classification

## 1. Mục tiêu dự án

Xây dựng một hệ thống **Deep Learning** để **phân loại HTTP request** thành:
- `normal`
- `attack` (SQLi, XSS, RCE, LFI, Path Traversal, …)

Hệ thống được thiết kế để:
- Hiểu **ngữ nghĩa (semantic)** của payload
- Phát hiện **zero-day / unknown attack**
- **Tích hợp trực tiếp với WAF** (inline hoặc async)
- **Mở rộng multi-model** (ML → DL → LLM)

---

## 2. Phạm vi & định hướng

### In scope
- Phân loại request dựa trên **raw HTTP data**
- Deep Learning (Transformer-based)
- Hỗ trợ inference realtime
- Chuẩn hoá input cho WAF

### Out of scope (giai đoạn đầu)
- Signature / rule-based detection
- Automated blocking logic
- Full traffic replay system

---

## 3. Kiến trúc tổng thể

HTTP Request  
↓  
Preprocessing & Normalization  
↓  
Feature Layer (optional ML)  
↓  
Deep Learning Semantic Model  
↓  
Decision (attack / normal)  
↓  
WAF Action (block / allow / challenge)

---

## 4. Nguyên tắc thiết kế

1. **Semantic-first**
   - Không phụ thuộc keyword thuần túy
   - Hiểu ngữ cảnh, thứ tự, cấu trúc payload

2. **Model-agnostic**
   - Không lock vào 1 model
   - Có thể chạy song song nhiều model

3. **WAF-friendly**
   - Latency thấp
   - Có thể chạy inline hoặc async
   - Output rõ ràng, dễ tích hợp

4. **Production-ready**
   - Versioning model
   - Reproducible training
   - Monitoring & logging

---

## 5. Input dữ liệu (chuẩn hoá cho DL)

### HTTP Request Schema

METHOD [SEP]  
PATH [SEP]  
QUERY_STRING [SEP]  
HEADERS [SEP]  
BODY

---

## 6. Nhãn (Label)

0 – normal  
1 – attack  

(Optional – multi-class):
- SQLi
- XSS
- RCE
- LFI
- Scanner / Bot

---

## 7. Model đề xuất

- Transformer encoder (DistilBERT / BERT / RoBERTa)
- BiLSTM / GRU (lightweight)
- ML truyền thống (filter)
- LLM (offline reasoning)

---

## 8. Cấu trúc project

waf-attack-detection/
├── data/
├── preprocessing/
├── models/
│   ├── transformer/
│   ├── lstm/
│   └── ml/
├── serving/
├── evaluation/
├── configs/
├── scripts/
└── README.md

---

## 9. API Output

{
  "is_attack": true,
  "confidence": 0.97,
  "model": "distilbert-v1",
  "attack_type": "SQLI"
}

---

## 10. Metric đánh giá

- Precision
- Recall
- F1-score
- ROC-AUC
- Latency

---

## 11. Training strategy

- Balanced dataset
- Data augmentation
- Hard-negative mining
- Continuous retraining from WAF logs

---

## 12. Deployment

- Inline: DistilBERT + ONNX + gRPC
- Async: Full Transformer / LLM

---

## 13. Nguyên tắc quan trọng

ML không hiểu semantic  
Rule không phát hiện zero-day  
Transformer là core semantic engine  
LLM dùng cho reasoning & explain
