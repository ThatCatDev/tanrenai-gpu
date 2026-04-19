package runner

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// ---- ParseSSEStream ----

func TestParseSSEStream_SingleChunk(t *testing.T) {
	chunk := api.ChatCompletionChunk{
		ID:    "cmpl-1",
		Model: "testmodel",
		Choices: []api.ChunkChoice{
			{Delta: api.MessageDelta{Content: "Hello"}},
		},
	}
	data, _ := json.Marshal(chunk)
	sse := "data: " + string(data) + "\n\ndata: [DONE]\n\n"

	ch := ParseSSEStream(strings.NewReader(sse))

	var events []StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}

	if len(events) != 2 {
		t.Fatalf("len(events) = %d, want 2", len(events))
	}
	if events[0].Chunk == nil {
		t.Fatal("events[0].Chunk is nil")
	}
	if events[0].Chunk.ID != "cmpl-1" {
		t.Errorf("chunk ID = %q, want %q", events[0].Chunk.ID, "cmpl-1")
	}
	if !events[1].Done {
		t.Error("events[1].Done should be true")
	}
}

func TestParseSSEStream_MultipleChunks(t *testing.T) {
	makeChunk := func(id, content string) string {
		chunk := api.ChatCompletionChunk{
			ID: id,
			Choices: []api.ChunkChoice{
				{Delta: api.MessageDelta{Content: content}},
			},
		}
		data, _ := json.Marshal(chunk)

		return "data: " + string(data) + "\n\n"
	}

	sse := makeChunk("1", "Hello") + makeChunk("2", " World") + "data: [DONE]\n\n"
	ch := ParseSSEStream(strings.NewReader(sse))

	var events []StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}

	// 2 content chunks + 1 done
	if len(events) != 3 {
		t.Fatalf("len(events) = %d, want 3", len(events))
	}
	if events[0].Chunk.Choices[0].Delta.Content != "Hello" {
		t.Errorf("chunk[0] content = %q, want %q", events[0].Chunk.Choices[0].Delta.Content, "Hello")
	}
	if events[1].Chunk.Choices[0].Delta.Content != " World" {
		t.Errorf("chunk[1] content = %q, want %q", events[1].Chunk.Choices[0].Delta.Content, " World")
	}
	if !events[2].Done {
		t.Error("last event should be done")
	}
}

func TestParseSSEStream_SkipsNonDataLines(t *testing.T) {
	chunk := api.ChatCompletionChunk{
		ID: "cmpl-skip",
		Choices: []api.ChunkChoice{
			{Delta: api.MessageDelta{Content: "hi"}},
		},
	}
	data, _ := json.Marshal(chunk)

	// Mix in comment lines, blank lines, event: lines
	sse := ": this is a comment\n\nevent: message\n\ndata: " + string(data) + "\n\ndata: [DONE]\n\n"
	ch := ParseSSEStream(strings.NewReader(sse))

	var events []StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}

	if len(events) != 2 {
		t.Fatalf("len(events) = %d, want 2 (1 chunk + done)", len(events))
	}
	if events[0].Chunk == nil || events[0].Chunk.ID != "cmpl-skip" {
		t.Errorf("unexpected first event: %+v", events[0])
	}
}

func TestParseSSEStream_BadJSON(t *testing.T) {
	sse := "data: {invalid json}\n\n"
	ch := ParseSSEStream(strings.NewReader(sse))

	var events []StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}

	if len(events) != 1 {
		t.Fatalf("len(events) = %d, want 1 (error event)", len(events))
	}
	if events[0].Err == nil {
		t.Error("expected error event for bad JSON")
	}
}

func TestParseSSEStream_EmptyStream(t *testing.T) {
	ch := ParseSSEStream(strings.NewReader(""))

	var events []StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}

	if len(events) != 0 {
		t.Errorf("len(events) = %d, want 0 for empty stream", len(events))
	}
}

func TestParseSSEStream_DoneStopsProcessing(t *testing.T) {
	chunk := api.ChatCompletionChunk{
		ID: "after-done",
		Choices: []api.ChunkChoice{
			{Delta: api.MessageDelta{Content: "should not appear"}},
		},
	}
	data, _ := json.Marshal(chunk)

	// DONE comes before another chunk
	sse := "data: [DONE]\n\ndata: " + string(data) + "\n\n"
	ch := ParseSSEStream(strings.NewReader(sse))

	var events []StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}

	// Only the [DONE] event; post-DONE chunks are never sent
	if len(events) != 1 {
		t.Fatalf("len(events) = %d, want 1 (only DONE)", len(events))
	}
	if !events[0].Done {
		t.Error("expected DONE event")
	}
}

// ---- AccumulateResponse ----

func makeEventsChan(events []StreamEvent) <-chan StreamEvent {
	ch := make(chan StreamEvent, len(events))
	for _, ev := range events {
		ch <- ev
	}
	close(ch)

	return ch
}

func TestAccumulateResponse_SimpleText(t *testing.T) {
	fr := "stop"
	events := []StreamEvent{
		{Chunk: &api.ChatCompletionChunk{
			ID:    "id1",
			Model: "mymodel",
			Choices: []api.ChunkChoice{
				{Delta: api.MessageDelta{Role: "assistant", Content: "Hello"}},
			},
		}},
		{Chunk: &api.ChatCompletionChunk{
			ID: "id1",
			Choices: []api.ChunkChoice{
				{Delta: api.MessageDelta{Content: " World"}, FinishReason: &fr},
			},
		}},
		{Done: true},
	}

	resp, err := AccumulateResponse(makeEventsChan(events))
	if err != nil {
		t.Fatalf("AccumulateResponse error: %v", err)
	}
	if resp.ID != "id1" {
		t.Errorf("ID = %q, want %q", resp.ID, "id1")
	}
	if resp.Model != "mymodel" {
		t.Errorf("Model = %q, want %q", resp.Model, "mymodel")
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("len(Choices) = %d, want 1", len(resp.Choices))
	}
	if resp.Choices[0].Message.Content != "Hello World" {
		t.Errorf("Content = %q, want %q", resp.Choices[0].Message.Content, "Hello World")
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("Role = %q, want %q", resp.Choices[0].Message.Role, "assistant")
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.Choices[0].FinishReason, "stop")
	}
}

func TestAccumulateResponse_DefaultRoleAssistant(t *testing.T) {
	events := []StreamEvent{
		{Chunk: &api.ChatCompletionChunk{
			Choices: []api.ChunkChoice{
				{Delta: api.MessageDelta{Content: "hi"}},
			},
		}},
		{Done: true},
	}

	resp, err := AccumulateResponse(makeEventsChan(events))
	if err != nil {
		t.Fatalf("AccumulateResponse error: %v", err)
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("Role = %q, want 'assistant'", resp.Choices[0].Message.Role)
	}
}

func TestAccumulateResponse_FinishReasonDefault(t *testing.T) {
	events := []StreamEvent{
		{Chunk: &api.ChatCompletionChunk{
			Choices: []api.ChunkChoice{
				{Delta: api.MessageDelta{Content: "hi"}},
			},
		}},
		{Done: true},
	}

	resp, err := AccumulateResponse(makeEventsChan(events))
	if err != nil {
		t.Fatalf("AccumulateResponse error: %v", err)
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want 'stop'", resp.Choices[0].FinishReason)
	}
}

func TestAccumulateResponse_ToolCalls(t *testing.T) {
	tc := "tool_calls"
	events := []StreamEvent{
		{Chunk: &api.ChatCompletionChunk{
			ID: "tc-1",
			Choices: []api.ChunkChoice{
				{Delta: api.MessageDelta{
					ToolCalls: []api.ToolCallDelta{
						{Index: 0, ID: "call-1", Type: "function", Function: &api.ToolCallFunction{Name: "my_func", Arguments: `{"key`}},
					},
				}},
			},
		}},
		{Chunk: &api.ChatCompletionChunk{
			ID: "tc-1",
			Choices: []api.ChunkChoice{
				{Delta: api.MessageDelta{
					ToolCalls: []api.ToolCallDelta{
						{Index: 0, Function: &api.ToolCallFunction{Arguments: `": "val"}`}},
					},
				}, FinishReason: &tc},
			},
		}},
		{Done: true},
	}

	resp, err := AccumulateResponse(makeEventsChan(events))
	if err != nil {
		t.Fatalf("AccumulateResponse error: %v", err)
	}
	if len(resp.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(resp.Choices[0].Message.ToolCalls))
	}
	tc0 := resp.Choices[0].Message.ToolCalls[0]
	if tc0.ID != "call-1" {
		t.Errorf("ToolCall.ID = %q, want %q", tc0.ID, "call-1")
	}
	if tc0.Function.Name != "my_func" {
		t.Errorf("ToolCall.Function.Name = %q, want %q", tc0.Function.Name, "my_func")
	}
	wantArgs := `{"key": "val"}`
	if tc0.Function.Arguments != wantArgs {
		t.Errorf("ToolCall.Function.Arguments = %q, want %q", tc0.Function.Arguments, wantArgs)
	}
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, want 'tool_calls'", resp.Choices[0].FinishReason)
	}
}

func TestAccumulateResponse_ToolCallsDefaultFinishReason(t *testing.T) {
	// No explicit finish_reason set in stream — should infer "tool_calls"
	events := []StreamEvent{
		{Chunk: &api.ChatCompletionChunk{
			Choices: []api.ChunkChoice{
				{Delta: api.MessageDelta{
					ToolCalls: []api.ToolCallDelta{
						{Index: 0, ID: "call-2", Type: "function", Function: &api.ToolCallFunction{Name: "fn", Arguments: `{}`}},
					},
				}},
			},
		}},
		{Done: true},
	}

	resp, err := AccumulateResponse(makeEventsChan(events))
	if err != nil {
		t.Fatalf("AccumulateResponse error: %v", err)
	}
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, want 'tool_calls'", resp.Choices[0].FinishReason)
	}
}

func TestAccumulateResponse_ErrorEvent(t *testing.T) {
	events := []StreamEvent{
		{Chunk: &api.ChatCompletionChunk{
			Choices: []api.ChunkChoice{
				{Delta: api.MessageDelta{Content: "partial"}},
			},
		}},
		{Err: &parseError{"stream error"}},
	}

	_, err := AccumulateResponse(makeEventsChan(events))
	if err == nil {
		t.Fatal("expected error from error event")
	}
}

func TestAccumulateResponse_NilChunkSkipped(t *testing.T) {
	events := []StreamEvent{
		{Chunk: nil}, // nil chunk — should be skipped
		{Chunk: &api.ChatCompletionChunk{
			ID: "cmpl-ok",
			Choices: []api.ChunkChoice{
				{Delta: api.MessageDelta{Content: "hello"}},
			},
		}},
		{Done: true},
	}

	resp, err := AccumulateResponse(makeEventsChan(events))
	if err != nil {
		t.Fatalf("AccumulateResponse error: %v", err)
	}
	if resp.ID != "cmpl-ok" {
		t.Errorf("ID = %q, want %q", resp.ID, "cmpl-ok")
	}
	if resp.Choices[0].Message.Content != "hello" {
		t.Errorf("Content = %q, want %q", resp.Choices[0].Message.Content, "hello")
	}
}

func TestAccumulateResponse_EmptyStream(t *testing.T) {
	events := []StreamEvent{}

	resp, err := AccumulateResponse(makeEventsChan(events))
	if err != nil {
		t.Fatalf("AccumulateResponse error: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("Role = %q, want 'assistant'", resp.Choices[0].Message.Role)
	}
}

func TestAccumulateResponse_Object(t *testing.T) {
	events := []StreamEvent{
		{Chunk: &api.ChatCompletionChunk{
			Choices: []api.ChunkChoice{
				{Delta: api.MessageDelta{Content: "test"}},
			},
		}},
		{Done: true},
	}

	resp, err := AccumulateResponse(makeEventsChan(events))
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if resp.Object != "chat.completion" {
		t.Errorf("Object = %q, want %q", resp.Object, "chat.completion")
	}
}

// parseError is a simple error type for testing.
type parseError struct{ msg string }

func (e *parseError) Error() string { return e.msg }
