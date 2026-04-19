package runner

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"

	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// StreamEvent represents a parsed SSE event from llama-server.
type StreamEvent struct {
	Chunk *api.ChatCompletionChunk
	Done  bool
	Err   error
}

// ParseSSEStream reads an SSE stream and sends parsed events to a channel.
// The channel is closed when the stream ends or an error occurs.
func ParseSSEStream(r io.Reader) <-chan StreamEvent {
	ch := make(chan StreamEvent)
	go func() {
		defer close(ch)
		scanner := bufio.NewScanner(r)

		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")

			if data == "[DONE]" {
				ch <- StreamEvent{Done: true}

				return
			}

			var chunk api.ChatCompletionChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				ch <- StreamEvent{Err: err}

				return
			}
			ch <- StreamEvent{Chunk: &chunk}
		}

		if err := scanner.Err(); err != nil {
			ch <- StreamEvent{Err: err}
		}
	}()

	return ch
}

// AccumulateResponse collects streaming chunks into a complete ChatCompletionResponse.
// It accumulates content and tool call deltas from the stream.
func AccumulateResponse(events <-chan StreamEvent) (*api.ChatCompletionResponse, error) {
	var (
		content      strings.Builder
		role         string
		model        string
		id           string
		finishReason string
		toolCalls    []api.ToolCall
		toolArgBuf   = make(map[int]*strings.Builder) // index -> accumulated arguments
	)

	for ev := range events {
		if ev.Err != nil {
			return nil, ev.Err
		}
		if ev.Done {
			break
		}
		if ev.Chunk == nil {
			continue
		}

		if id == "" {
			id = ev.Chunk.ID
		}
		if model == "" {
			model = ev.Chunk.Model
		}

		for _, choice := range ev.Chunk.Choices {
			if choice.Delta.Role != "" {
				role = choice.Delta.Role
			}
			// Capture finish_reason from the stream (set on last chunk)
			if choice.FinishReason != nil {
				finishReason = *choice.FinishReason
			}
			if choice.Delta.Content != "" {
				content.WriteString(choice.Delta.Content)
			}

			for _, tcd := range choice.Delta.ToolCalls {
				// Grow the toolCalls slice if needed
				for len(toolCalls) <= tcd.Index {
					toolCalls = append(toolCalls, api.ToolCall{})
				}
				if tcd.ID != "" {
					toolCalls[tcd.Index].ID = tcd.ID
				}
				if tcd.Type != "" {
					toolCalls[tcd.Index].Type = tcd.Type
				}
				if tcd.Function != nil {
					if tcd.Function.Name != "" {
						toolCalls[tcd.Index].Function.Name = tcd.Function.Name
					}
					if tcd.Function.Arguments != "" {
						if toolArgBuf[tcd.Index] == nil {
							toolArgBuf[tcd.Index] = &strings.Builder{}
						}
						toolArgBuf[tcd.Index].WriteString(tcd.Function.Arguments)
					}
				}
			}
		}
	}

	// Finalize accumulated tool call arguments
	for idx, buf := range toolArgBuf {
		if idx < len(toolCalls) {
			toolCalls[idx].Function.Arguments = buf.String()
		}
	}

	if role == "" {
		role = "assistant"
	}

	msg := api.Message{
		Role:    role,
		Content: content.String(),
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	// Use the actual finish_reason from the stream; fall back to inference
	if finishReason == "" {
		finishReason = "stop"
		if len(toolCalls) > 0 {
			finishReason = "tool_calls"
		}
	}

	return &api.ChatCompletionResponse{
		ID:     id,
		Object: "chat.completion",
		Model:  model,
		Choices: []api.Choice{
			{
				Index:        0,
				Message:      msg,
				FinishReason: finishReason,
			},
		},
	}, nil
}
