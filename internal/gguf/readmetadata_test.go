package gguf

import (
	"encoding/binary"
	"os"
	"testing"
)

// TestReadMetadata_FileNotFound ensures ReadMetadata propagates OS errors.
func TestReadMetadata_FileNotFound(t *testing.T) {
	_, err := ReadMetadata("/nonexistent/path/model.gguf")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

// TestReadMetadata_FromFile tests the public ReadMetadata function via a real
// temporary file built with the ggufBuilder helper from reader_test.go.
func TestReadMetadata_FromFile(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "llama")
	b.addKV("general.name", valueTypeString, "TestModel-7B")
	b.addKV("llama.context_length", valueTypeUint32, uint32(4096))
	b.addKV("tokenizer.chat_template", valueTypeString, "{% for m in messages %}{{ m }}{% endfor %}")

	reader := b.build()
	data := make([]byte, reader.Len())
	_, _ = reader.Read(data)

	// Write to a temp file
	f, err := os.CreateTemp(t.TempDir(), "*.gguf")
	if err != nil {
		t.Fatalf("create temp file: %v", err)
	}
	defer f.Close()
	if _, err := f.Write(data); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close temp file: %v", err)
	}

	meta, err := ReadMetadata(f.Name())
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	if meta.General.Architecture != "llama" {
		t.Errorf("architecture = %q, want %q", meta.General.Architecture, "llama")
	}
	if meta.General.Name != "TestModel-7B" {
		t.Errorf("name = %q, want %q", meta.General.Name, "TestModel-7B")
	}
	if meta.Architecture.ContextLength != 4096 {
		t.Errorf("context_length = %d, want 4096", meta.Architecture.ContextLength)
	}
	if meta.Tokenizer.ChatTemplate == "" {
		t.Error("chat_template should not be empty")
	}
}

// TestReadMetadata_V2_FromFile tests v2 GGUF reading from a real file.
func TestReadMetadata_V2_FromFile(t *testing.T) {
	b := newBuilder(2)
	b.addKV("general.architecture", valueTypeString, "qwen2")
	b.addKV("general.name", valueTypeString, "Qwen2-7B")

	reader := b.build()
	data := make([]byte, reader.Len())
	_, _ = reader.Read(data)

	f, err := os.CreateTemp(t.TempDir(), "*.gguf")
	if err != nil {
		t.Fatalf("create temp file: %v", err)
	}
	defer f.Close()
	if _, err := f.Write(data); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close temp file: %v", err)
	}

	meta, err := ReadMetadata(f.Name())
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	if meta.Version != 2 {
		t.Errorf("version = %d, want 2", meta.Version)
	}
	if meta.General.Architecture != "qwen2" {
		t.Errorf("architecture = %q, want %q", meta.General.Architecture, "qwen2")
	}
}

// TestReadMetadata_UnknownValueType verifies we get an error for unknown types.
func TestReadMetadata_UnknownValueType(t *testing.T) {
	// Manually construct a GGUF with an unknown value type (99).
	var buf []byte
	writeU32 := func(v uint32) {
		b := make([]byte, 4)
		binary.LittleEndian.PutUint32(b, v)
		buf = append(buf, b...)
	}
	writeU64 := func(v uint64) {
		b := make([]byte, 8)
		binary.LittleEndian.PutUint64(b, v)
		buf = append(buf, b...)
	}
	writeStr := func(s string) {
		writeU64(uint64(len(s)))
		buf = append(buf, []byte(s)...)
	}

	// Magic + version 3
	writeU32(ggufMagic)
	writeU32(3)
	writeU64(0) // tensor count
	writeU64(1) // 1 KV pair
	writeStr("bad.key")
	writeU32(99) // unknown value type

	f, err := os.CreateTemp(t.TempDir(), "*.gguf")
	if err != nil {
		t.Fatalf("create temp file: %v", err)
	}
	defer f.Close()
	if _, err := f.Write(buf); err != nil {
		t.Fatalf("write: %v", err)
	}
	_ = f.Close()

	_, err = ReadMetadata(f.Name())
	if err == nil {
		t.Fatal("expected error for unknown value type")
	}
}
