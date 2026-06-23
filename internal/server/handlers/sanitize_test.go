package handlers

import (
	"testing"

	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

func TestSanitizeRepairsTruncatedArguments(t *testing.T) {
	req := &api.ChatCompletionRequest{
		Messages: []api.Message{
			{Role: "user", Content: "weather?"},
			{Role: "assistant", ToolCalls: []api.ToolCall{
				{ID: "call_1", Type: "function", Function: api.ToolCallFunction{
					Name: "get_weather", Arguments: `{"city":"San Franci`, // truncated
				}},
			}},
		},
	}
	if n := sanitizeToolCallArguments(req); n != 1 {
		t.Fatalf("repaired = %d, want 1", n)
	}
	if got := req.Messages[1].ToolCalls[0].Function.Arguments; got != "{}" {
		t.Errorf("arguments = %q, want %q", got, "{}")
	}
}

func TestSanitizeLeavesValidArguments(t *testing.T) {
	valid := `{"city":"San Francisco","unit":"celsius"}`
	req := &api.ChatCompletionRequest{
		Messages: []api.Message{
			{Role: "assistant", ToolCalls: []api.ToolCall{
				{Function: api.ToolCallFunction{Name: "get_weather", Arguments: valid}},
			}},
		},
	}
	if n := sanitizeToolCallArguments(req); n != 0 {
		t.Fatalf("repaired = %d, want 0 (valid JSON must be untouched)", n)
	}
	if got := req.Messages[0].ToolCalls[0].Function.Arguments; got != valid {
		t.Errorf("valid arguments were modified: %q", got)
	}
}

func TestSanitizeNormalizesEmptyArgumentsSilently(t *testing.T) {
	req := &api.ChatCompletionRequest{
		Messages: []api.Message{
			{Role: "assistant", ToolCalls: []api.ToolCall{
				{Function: api.ToolCallFunction{Name: "now", Arguments: ""}},
			}},
		},
	}
	// Empty -> "{}" is benign normalization, not counted as a repair.
	if n := sanitizeToolCallArguments(req); n != 0 {
		t.Fatalf("repaired = %d, want 0 for empty-string normalization", n)
	}
	if got := req.Messages[0].ToolCalls[0].Function.Arguments; got != "{}" {
		t.Errorf("empty arguments = %q, want %q", got, "{}")
	}
}
