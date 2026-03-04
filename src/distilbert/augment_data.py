import pandas as pd
import random
import urllib.parse
import string
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
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

def read_raw_data(txt_path, is_attack):
    payloads = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[2:]
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(': ', 1)
        if len(parts) == 2:
            prefix = parts[0]
            payload = parts[1]
            if is_attack:
                label_str = categorize_attack(prefix, payload)
            else:
                label_str = 'normal'
        else:
            payload = line
            label_str = categorize_attack('', payload) if is_attack else 'normal'
            
        payloads.append((payload, label_str))
    return payloads

def get_random_param():
    return random.choice(['id', 'page', 'user', 'search', 'q', 'query', 'file', 'dir', 'path', 'name', 'filter', 'sort'])

def get_random_string(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def get_random_path():
    return random.choice(['/assets/', '/images/', '/css/', '/api/v1/users', '/login', '/admin', '/index.php', '/search', '/api/docs'])

def modify_sqli(payload):
    # Swap '1'='1' style with variables
    if random.random() > 0.5:
        a = get_random_string(1)
        payload = payload.replace("1'='1", f"{a}'='{a}")
        payload = payload.replace("1=1", f"{get_random_string(1)}={get_random_string(1)}")
    
    keywords = ['UNION', 'SELECT', 'DROP', 'INSERT', 'UPDATE', 'DELETE', 'AND', 'OR']
    for k in keywords:
        if k in payload and random.random() > 0.5:
            # Change case pseudo-randomly
            if random.random() > 0.5:
                payload = payload.replace(k, k.lower())
            else:
                payload = payload.replace(k, k.capitalize())
    return payload

def modify_xss(payload):
    # Swap alert(1) with confirm or prompt randomly
    if random.random() > 0.5:
        payload = payload.replace("alert(1)", random.choice(["confirm(1)", "prompt(1)", "alert('XSS')"]))
    # Change tag closures randomly
    if random.random() > 0.5:
        payload = payload.replace("<script>", random.choice(["<SeRiPt>", "<scr ipt>", " <script >"]))
    return payload

def modify_lfi_traversal(payload):
    # Vary the depth
    if "../" in payload:
        depth = random.randint(1, 10)
        payload = payload.replace("../" * payload.count("../"), "../" * depth)
    # Vary the target file
    payload = payload.replace("/etc/passwd", random.choice(["/etc/passwd", "/etc/shadow", "/Windows/win.ini", "/boot.ini"]))
    return payload

def augment_payload(payload, label_str):
    # 1. Base class-specific mutations
    if label_str == 'sqli':
        payload = modify_sqli(payload)
    elif label_str == 'xss':
        payload = modify_xss(payload)
    elif label_str in ['lfi', 'traversal']:
        payload = modify_lfi_traversal(payload)
        
    # 2. General mutations
    chance = random.random()
    if chance < 0.2:
        # Wrap in a parameter (id=payload)
        payload = f"{get_random_param()}={payload}"
    elif chance < 0.4:
        # Wrap in a path + parameter (/login?user=payload)
        payload = f"{get_random_path()}?{get_random_param()}={payload}"
    elif chance < 0.6:
        # URL encode
        payload = urllib.parse.quote_plus(payload)
    elif chance < 0.8:
        # Just append some random junk at the end
        if label_str == 'normal':
             payload = payload + f"&{get_random_param()}={get_random_string()}"
        else:
             payload = payload + " -- " + get_random_string(4)
             
    # Normal payloads might just be changed entirely to a new path to create diversity
    if label_str == 'normal' and random.random() < 0.3:
        payload = f"{get_random_path()}?{get_random_param()}={get_random_string()}"
        
    return payload

def generate_augmented_dataset(target_size=20000):
    logger.info("Reading raw files...")
    normal_data = read_raw_data('../../data/normal.txt', is_attack=False)
    attack_data = read_raw_data('../../data/attack.txt', is_attack=True)
    
    all_raw = normal_data + attack_data
    logger.info(f"Loaded {len(all_raw)} raw base templates.")
    
    augmented_rows = []
    
    # ensure we keep the raw ones
    for p, l in all_raw:
        augmented_rows.append({'text': p, 'label_str': l, 'label': CLASS_MAP[l]})
        
    # Generate the rest
    needed = target_size - len(augmented_rows)
    logger.info(f"Generating {needed} synthetic payloads...")
    
    for _ in range(needed):
        base_payload, label_str = random.choice(all_raw)
        new_payload = augment_payload(base_payload, label_str)
        augmented_rows.append({'text': new_payload, 'label_str': label_str, 'label': CLASS_MAP[label_str]})
        
    df = pd.DataFrame(augmented_rows)
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    output_path = '../../data/augmented_dataset.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"Saved augmented dataset ({len(df)} samples) to {output_path}")
    
    logger.info("Class distribution in augmented dataset:")
    logger.info(df['label_str'].value_counts())

if __name__ == "__main__":
    generate_augmented_dataset(target_size=20000)
