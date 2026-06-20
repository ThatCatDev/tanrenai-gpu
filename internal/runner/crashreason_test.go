package runner

import "testing"

func TestExtractCrashReasonPrefersErrorOverBacktrace(t *testing.T) {
	// Realistic shape: operational logs, the cause line, then a backtrace.
	dump := `slot update_slots: id  0 | task 12 | n_past = 4096
CUDA error: out of memory
  current device: 0, in function ggml_cuda_pool_malloc at ggml-cuda.cu:512
/usr/local/lib/libggml-cuda.so(+0x12abc)[0x7fb11fbcba4c]
/usr/local/lib/libllama-server-impl.so(_ZN12server_queue10start_loopEl+0x221)[0x7fb11fc468c1]
/usr/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x80)[0x7fb11f61ae40]`

	got := extractCrashReason(dump)
	if got != "CUDA error: out of memory" {
		t.Errorf("got %q, want the CUDA error line (not a backtrace frame)", got)
	}
}

func TestExtractCrashReasonFallsBackToHeader(t *testing.T) {
	// No recognizable signal line, only frames + a header — return the header,
	// never a raw stack frame.
	dump := `llama-server received signal 11
/usr/local/lib/libfoo.so(+0x1)[0x7f0000000001]
/usr/local/lib/libbar.so(+0x2)[0x7f0000000002]`
	got := extractCrashReason(dump)
	if got != "llama-server received signal 11" {
		t.Errorf("got %q, want the header line", got)
	}
}

func TestExtractCrashReasonEmpty(t *testing.T) {
	if got := extractCrashReason(""); got != "" {
		t.Errorf("got %q, want empty", got)
	}
}
