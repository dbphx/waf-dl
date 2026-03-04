package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/waf-dl/application/go/utils"
)

type WafResponse struct {
	Status string `json:"status"` // "NORMAL" or "ATTACK"
	Type   string `json:"type"`   // "normal", "sqli", "xss", etc
}

func main() {
	predictor, err := utils.NewPredictor(
		"models/distilbert_waf.onnx",
		"models/tokenizer/tokenizer.json",
	)
	if err != nil {
		log.Fatalf("Failed to initialize predictor: %v", err)
	}
	defer predictor.Close()

	fmt.Println("Model and ONNX session loaded successfully.")

	http.HandleFunc("/check", func(w http.ResponseWriter, r *http.Request) {
		// Parse form data from URL queries or POST body
		err := r.ParseForm()
		if err != nil {
			http.Error(w, "Unable to parse form", http.StatusBadRequest)
			return
		}

		args := make(map[string]string)
		for k, v := range r.Form {
			if len(v) > 0 {
				args[k] = v[0]
			}
		}

		label := predictor.PredictDL(args)

		status := "ATTACK"
		if strings.ToLower(label) == "normal" {
			status = "NORMAL"
		}

		resp := WafResponse{
			Status: status,
			Type:   label,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	fmt.Println("Server listening on :8080...")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
