package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"strings"

	"github.com/yalue/onnxruntime_go"
)

// A very simplified wordpiece tokenizer implementation for DistilBERT
// in a real application you would use a full go tokenizer library like
// github.com/sugarme/tokenizer or github.com/nlpodyssey/cybertron/pkg/tokenizers

type SimpleTokenizer struct {
	Vocab map[string]int
}

func LoadTokenizer(path string) (*SimpleTokenizer, error) {
	vocabData, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var vocab map[string]int
	if err := json.Unmarshal(vocabData, &vocab); err != nil {
		return nil, err
	}

	return &SimpleTokenizer{Vocab: vocab}, nil
}

func (t *SimpleTokenizer) Encode(text string, maxLength int) ([]int64, []int64) {
	// extremely simplified tokenization: lowercasing and character level fallback
	// This is ONLY for prototype logic illustration.
	text = strings.ToLower(text)

	inputIds := make([]int64, maxLength)
	attentionMask := make([]int64, maxLength)

	// CLS token
	inputIds[0] = 101
	attentionMask[0] = 1

	idx := 1
	for _, char := range text {
		if idx >= maxLength-1 {
			break
		}

		charStr := string(char)
		if token, exists := t.Vocab[charStr]; exists {
			inputIds[idx] = int64(token)
		} else {
			inputIds[idx] = 100 // UNK token
		}
		attentionMask[idx] = 1
		idx++
	}

	// SEP token
	inputIds[idx] = 102
	attentionMask[idx] = 1

	// padding
	for i := idx + 1; i < maxLength; i++ {
		inputIds[i] = 0
		attentionMask[i] = 0
	}

	return inputIds, attentionMask
}

func main() {
	// Initialize ONNX runtime
	onnxruntime_go.SetSharedLibraryPath(getSharedLibPath())
	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Error initializing ONNX runtime: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	// Load model
	modelPath := "../../models/deep_learning/distilbert_waf.onnx"

	// DistilBERT expects sequence of shape [batch_size, sequence_length]
	inputShape := []int64{1, 128}

	// Create inputs
	inputIdsData := make([]int64, 128)
	attentionMaskData := make([]int64, 128)

	inputIdsTensor, _ := onnxruntime_go.NewTensor(inputShape, inputIdsData)
	defer inputIdsTensor.Destroy()

	attentionMaskTensor, _ := onnxruntime_go.NewTensor(inputShape, attentionMaskData)
	defer attentionMaskTensor.Destroy()

	// Output tensor expected [1, 8] probabilities/logits
	outputShape := []int64{1, 8}
	outputData := make([]float32, 8)
	outputTensor, _ := onnxruntime_go.NewTensor(outputShape, outputData)
	defer outputTensor.Destroy()

	// Provide null for options since we want defaults, remove the explicit tensors from session init
	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		nil,
	)
	if err != nil {
		log.Fatalf("Error creating session: %v", err)
	}
	defer session.Destroy()

	// fall back to json if we exported it as json
	tokenizerJSONPath := "../../models/deep_learning/tokenizer/tokenizer.json"

	fmt.Println("Model and ONNX session loaded successfully.")
	fmt.Printf("Model path: %s\n", modelPath)
	fmt.Printf("Tokens configuration path: %s\n", tokenizerJSONPath)

	classes := []string{"normal", "sqli", "xss", "lfi", "rce", "other", "upload", "traversal"}

	payloads := []string{
		"GET /index.php",
		"GET /login?user=admin' OR '1'='1",
		"<script>alert(1)</script>",
		"../../../../etc/passwd",
	}

	fmt.Println("\nTesting Inference...")

	for _, payload := range payloads {
		// Mock tokenization for example execution
		log.Printf("Analyzing payload: %s", payload)

		// execute run
		err = session.Run(
			[]onnxruntime_go.Value{inputIdsTensor, attentionMaskTensor},
			[]onnxruntime_go.Value{outputTensor},
		)
		if err != nil {
			log.Fatalf("Run failed: %v", err)
		}

		// The outputData slice now contains the 8 logits
		fmt.Printf("Raw Logits: %v\n", outputData)

		// In a real app we apply Softmax/Argmax
		maxIdx := 0
		maxVal := outputData[0]
		for i, val := range outputData {
			if val > maxVal {
				maxVal = val
				maxIdx = i
			}
		}
		fmt.Printf("Prediction (mock data): %s\n\n", classes[maxIdx])
	}
}

func getSharedLibPath() string {
	// Helper to find the correct dynamic library for the host OS
	// For Mac: libonnxruntime.dylib, Linux: libonnxruntime.so
	return "/usr/local/lib/libonnxruntime.dylib"
}
