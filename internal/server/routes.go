package server

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	"github.com/ThatCatDev/tanrenai-gpu/internal/runner"
	"github.com/ThatCatDev/tanrenai-gpu/internal/server/handlers"
)

func (s *Server) registerRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /health", handlers.Health)
	mux.HandleFunc("GET /v1/version", handlers.Version)
	mux.HandleFunc("GET /v1/status", s.handleStatus)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("POST /api/load", s.handleLoadModel)
	mux.HandleFunc("POST /api/pull", s.handlePullModel)
	mux.HandleFunc("POST /tokenize", s.handleTokenize)
	mux.HandleFunc("POST /v1/embeddings", s.handleEmbeddings)

	// Fine-tuning endpoints (only active if training manager is set)
	if s.trainingManager != nil {
		ft := &handlers.FinetuneHandler{Manager: s.trainingManager}
		mux.HandleFunc("POST /v1/finetune/prepare", ft.Prepare)
		mux.HandleFunc("POST /v1/finetune/train", ft.Train)
		mux.HandleFunc("GET /v1/finetune/status/", ft.Status)
		mux.HandleFunc("POST /v1/finetune/merge", ft.Merge)
		mux.HandleFunc("GET /v1/finetune/runs", ft.ListRuns)
		mux.HandleFunc("DELETE /v1/finetune/runs/", ft.DeleteRun)
	}
}

// handleStatus reports the currently loaded model and its serving capacity:
// the total context window, the number of concurrent sequence slots
// (--parallel, ≈ simultaneous users), and the resulting per-user context. The
// platform uses `parallel` as the real capacity of a (potentially shared)
// instance. Returns 503 until a model is loaded.
func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	s.loadMu.Lock()
	loaded := s.loaded
	s.loadMu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	if loaded == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(map[string]any{"loaded": false})

		return
	}

	ctxPerUser := loaded.CtxSize
	if loaded.Parallel > 1 {
		ctxPerUser = loaded.CtxSize / loaded.Parallel
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"loaded":       true,
		"model":        loaded.Model,
		"ctx_size":     loaded.CtxSize,
		"parallel":     loaded.Parallel,
		"ctx_per_user": ctxPerUser,
	})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	h := &handlers.ModelsHandler{Store: s.store}
	h.ServeHTTP(w, r)
}

func (s *Server) wrapLoadFunc() func(ctx context.Context, model string) (*handlers.LoadResult, error) {
	return func(ctx context.Context, model string) (*handlers.LoadResult, error) {
		res, err := s.LoadModel(ctx, model)
		if err != nil {
			return nil, err
		}

		// Report the PER-USER window, not the total. With --parallel N the
		// total ctx is split across N slots, so a single client/session only
		// gets ctx_size/N. Clients budget their own context (and trigger
		// compaction) off this number — handing them the total made them
		// compact ~N× too late and overflow their slot.
		return &handlers.LoadResult{CtxSize: effectiveCtxPerUser(res.CtxSize, res.Parallel)}, nil
	}
}

// effectiveCtxPerUser is the context window a single session actually gets:
// the total context divided across parallel slots.
func effectiveCtxPerUser(ctxSize, parallel int) int {
	if parallel > 1 {
		return ctxSize / parallel
	}

	return ctxSize
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	h := &handlers.ChatHandler{
		GetRunner: func() runner.Runner { return s.runner },
		LoadFunc:  s.wrapLoadFunc(),
	}
	h.ServeHTTP(w, r)
}

func (s *Server) handleLoadModel(w http.ResponseWriter, r *http.Request) {
	h := &handlers.LoadHandler{LoadFunc: s.wrapLoadFunc()}
	h.ServeHTTP(w, r)
}

func (s *Server) handlePullModel(w http.ResponseWriter, r *http.Request) {
	h := &handlers.PullHandler{Store: s.store}
	h.ServeHTTP(w, r)
}

func (s *Server) handleTokenize(w http.ResponseWriter, r *http.Request) {
	h := &handlers.TokenizeHandler{
		GetRunner: func() runner.Runner { return s.runner },
	}
	h.ServeHTTP(w, r)
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	var baseURL string
	if s.embeddingRunner != nil {
		baseURL = s.embeddingRunner.BaseURL
	}
	h := &handlers.EmbeddingsHandler{EmbeddingBaseURL: baseURL}
	h.ServeHTTP(w, r)
}

// responseWriter wraps http.ResponseWriter to capture the status code.
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// Flush implements http.Flusher so streaming endpoints (SSE) work through the
// logging middleware.
func (rw *responseWriter) Flush() {
	if f, ok := rw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(wrapped, r)
		slog.Info("http request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", wrapped.statusCode,
			"latency_ms", time.Since(start).Milliseconds(),
		)
	})
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)

			return
		}
		next.ServeHTTP(w, r)
	})
}
