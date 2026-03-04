package utils

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"strings"

	"github.com/yalue/onnxruntime_go"
)

// WAFPredictor defines the interface for attack prediction
type WAFPredictor interface {
	PredictDL(args map[string]string) string
	Close()
}

// SimpleTokenizer is a very simplified wordpiece tokenizer implementation for DistilBERT.
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

// Predictor wraps the ONNX session and Tokenizer for clean inference bounds
type Predictor struct {
	session   *onnxruntime_go.DynamicAdvancedSession
	tokenizer *SimpleTokenizer
	classes   []string
	maxLength int
}

func NewPredictor(modelPath, vocabPath string) (WAFPredictor, error) {
	// Initialize ONNX runtime globally
	onnxruntime_go.SetSharedLibraryPath(getSharedLibPath())
	_ = onnxruntime_go.InitializeEnvironment()

	tokenizer, err := LoadTokenizer(vocabPath)
	if err != nil {
		// Mock gracefully if tokenizer is missing for the example execution
		tokenizer = &SimpleTokenizer{Vocab: make(map[string]int)}
	}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		nil,
	)
	if err != nil {
		return nil, err
	}

	return &Predictor{
		session:   session,
		tokenizer: tokenizer,
		classes:   []string{"normal", "sqli", "xss", "lfi", "rce", "other", "upload", "traversal"},
		maxLength: 128,
	}, nil
}

func (p *Predictor) Close() {
	if p.session != nil {
		p.session.Destroy()
	}
	_ = onnxruntime_go.DestroyEnvironment()
}

// PredictDL takes a map of request arguments and returns an attack class prediction
func (p *Predictor) PredictDL(args map[string]string) string {
	// Combine the map values into a single payload string
	var b strings.Builder
	for k, v := range args {
		b.WriteString(k)
		b.WriteString("=")
		b.WriteString(v)
		b.WriteString(" ")
	}
	payload := strings.TrimSpace(b.String())

	inputIdsData, attentionMaskData := p.tokenizer.Encode(payload, p.maxLength)

	inputShape := []int64{1, int64(p.maxLength)}
	inputIdsTensor, _ := onnxruntime_go.NewTensor(inputShape, inputIdsData)
	defer inputIdsTensor.Destroy()

	attentionMaskTensor, _ := onnxruntime_go.NewTensor(inputShape, attentionMaskData)
	defer attentionMaskTensor.Destroy()

	outputShape := []int64{1, 8}
	outputData := make([]float32, 8)
	outputTensor, _ := onnxruntime_go.NewTensor(outputShape, outputData)
	defer outputTensor.Destroy()

	err := p.session.Run(
		[]onnxruntime_go.Value{inputIdsTensor, attentionMaskTensor},
		[]onnxruntime_go.Value{outputTensor},
	)
	if err != nil {
		log.Printf("Inference failed: %v", err)
		return "error"
	}

	maxIdx := 0
	maxVal := outputData[0]
	for i, val := range outputData {
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}

	return p.classes[maxIdx]
}

func getSharedLibPath() string {
	// Helper to find the correct dynamic library for the host OS
	// For Mac: libonnxruntime.dylib, Linux: libonnxruntime.so
	return "/usr/local/lib/libonnxruntime.dylib"
}
