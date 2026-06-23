package handlers

import (
	"encoding/json"
	"log/slog"

	"github.com/ThatCatDev/tanrenai-gpu/pkg/api"
)

// sanitizeToolCallArguments repairs tool-call arguments in the inbound message
// history that aren't valid JSON, so llama-server's strict tool parser doesn't
// reject the whole request with a 500 while rendering the prompt.
//
// The failure mode this guards against: a streaming tool call gets truncated
// (the box crashed / the stream was cut before the arguments JSON closed), the
// client receives a partial tool call like `{"city":"San Franci`, and replays
// it in the next request's history. llama-server then fails to parse it
// ("unexpected end of input; expected '}'") and 500s the entire request — so a
// past hiccup poisons every subsequent turn of the conversation.
//
// We can't recover the lost content, so we log the raw value (for diagnosis)
// and substitute an empty JSON object: the request succeeds, the conversation
// continues, and the tool_call/tool_result pairing stays intact. Empty
// arguments ("") are normalized to "{}" silently — that's benign, not a fault.
//
// Returns the number of arguments that were actually repaired (invalid JSON);
// silent empty-string normalizations are not counted.
func sanitizeToolCallArguments(req *api.ChatCompletionRequest) int {
	repaired := 0
	for mi := range req.Messages {
		m := &req.Messages[mi]
		for ti := range m.ToolCalls {
			args := m.ToolCalls[ti].Function.Arguments
			if args == "" {
				m.ToolCalls[ti].Function.Arguments = "{}"
				continue
			}
			if json.Valid([]byte(args)) {
				continue
			}
			slog.Warn("repairing malformed tool-call arguments in request history",
				"role", m.Role,
				"message_index", mi,
				"tool", m.ToolCalls[ti].Function.Name,
				"tool_call_id", m.ToolCalls[ti].ID,
				"raw_arguments", args)
			m.ToolCalls[ti].Function.Arguments = "{}"
			repaired++
		}
	}
	return repaired
}
