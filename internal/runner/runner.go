package runner

import (
	"context"
	"io"

	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// Runner is the interface for model inference backends.
// The ProcessRunner (Stage 1) manages llama-server as a subprocess.
// A future DirectRunner could use CGo/purego bindings for in-process inference.
type Runner interface {
	// Load starts the runner with the given model.
	Load(ctx context.Context, modelPath string, opts Options) error

	// Health returns nil if the runner is ready to serve requests.
	Health(ctx context.Context) error

	// ChatCompletion performs a non-streaming chat completion.
	ChatCompletion(ctx context.Context, req *api.ChatCompletionRequest) (*api.ChatCompletionResponse, error)

	// ChatCompletionStream performs a streaming chat completion, writing SSE chunks to the writer.
	ChatCompletionStream(ctx context.Context, req *api.ChatCompletionRequest, w io.Writer) error

	// Tokenize returns the token count for the given text using the server's tokenizer.
	Tokenize(ctx context.Context, text string) (int, error)

	// ModelName returns the name/ID of the loaded model.
	ModelName() string

	// Close shuts down the runner and releases resources.
	Close() error
}
