package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/waf-dl/application/go/utils"
)

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

	// Example usage
	args := map[string]string{
		"query_user":   "admin' OR '1'='1",
		"body_comment": "<script>alert(1)</script>",
	}

	fmt.Printf("\nEvaluating arguments: %v\n", args)
	label := predictor.PredictDL(args)

	fmt.Printf("Prediction Result: %s\n", strings.ToUpper(label))
}
