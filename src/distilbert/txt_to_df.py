import pandas as pd
import re
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Same broader class mapping as before to maintain some dimension normalization
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

def categorize_attack(attack_type, payload=""):
    attack_type = attack_type.lower()
    payload_lower = payload.lower()
    
    if 'sql' in attack_type:
        return 'sqli'
    elif 'xss' in attack_type or 'cross-site' in attack_type:
        return 'xss'
    elif 'lfi' in attack_type or 'local file inclusion' in attack_type:
        return 'lfi'
    elif 'rce' in attack_type or 'command injection' in attack_type or 'execution' in attack_type:
        return 'rce'
    elif 'traversal' in attack_type or 'directory traversal' in attack_type:
        return 'traversal'
    elif 'upload' in attack_type:
        return 'upload'

    # Heuristics based on payload contents if prefix is generic (e.g. Attack_PDF)
    if any(k in payload_lower for k in ['script', 'alert(', 'onerror=', 'onload=', 'prompt(', 'confirm(', 'svg', 'javascript:', 'onmouseover=', 'onmouseenter=', 'onfocus=', 'onauxclick=', 'onpointer']):
        return 'xss'
    if any(k in payload_lower for k in ['select ', 'union ', 'waitfor ', 'sleep(', 'or 1=1', "or '1'='1", 'drop table', 'information_schema', 'json_extract']):
        return 'sqli'
    if any(k in payload_lower for k in ['/etc/passwd', 'boot.ini', 'win.ini']):
        return 'lfi'
    if any(k in payload_lower for k in ['../', '..%2f', '..%c0%af', '..\\']):
        return 'traversal'
    if any(k in payload_lower for k in ['; cat', '| ls', '$(whoami)', '`id`', 'wget ', 'curl ', 'exec cmd', 'ping ', 'whoami', '| set /a']):
        return 'rce'
        
    return 'other'

def parse_txt_to_df(txt_path, is_attack):
    rows = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        # skip first 2 lines
        lines = f.readlines()[2:]
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(': ', 1)
        if len(parts) == 2:
            # Ex: "1. SQL Injection (Basic): id=1' OR ..."
            prefix = parts[0]
            payload = parts[1]
            
            # The prefix might be "1. Category (Sub)" or "Attack_PDF_1"
            # It's dirty so let's classify it based on string matching
            if is_attack:
                label_str = categorize_attack(prefix, payload)
            else:
                label_str = 'normal'
            
        else:
            # Not nicely formatted with colon
            payload = line
            label_str = categorize_attack('', payload) if is_attack else 'normal'
            
        rows.append({'text': payload, 'label_str': label_str, 'label': CLASS_MAP[label_str]})
        
    return pd.DataFrame(rows)

def prepare_hybrid_dataset():
    logger.info("Parsing normal.txt")
    df_normal_txt = parse_txt_to_df('../../data/normal.txt', is_attack=False)
    
    logger.info("Parsing attack.txt")
    df_attack_txt = parse_txt_to_df('../../data/attack.txt', is_attack=True)
    
    df_hybrid = pd.concat([df_normal_txt, df_attack_txt], ignore_index=True)
    df_hybrid.to_csv('../../data/hybrid_dataset.csv', index=False)
    logger.info("Saved hybrid dataset combining texts to hybrid_dataset.csv")
    
    logger.info("Class distribution in hybrid dataset:")
    logger.info(df_hybrid['label_str'].value_counts())
    
    return df_hybrid

if __name__ == "__main__":
    prepare_hybrid_dataset()
