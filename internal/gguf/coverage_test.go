package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// TestReadMetadata_AllNumericTypes exercises all numeric value types for
// the arch prefix path (testing mapUint32, mapUint64, mapFloat32 coercions).
func TestReadMetadata_AllNumericArchTypes(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "llama")

	// uint16 coercion → mapUint64
	b.addKV("llama.context_length", valueTypeUint16, uint16(8192))
	// uint8 → mapUint64
	b.addKV("llama.embedding_length", valueTypeUint8, uint8(32))
	// int32 → mapUint32
	b.addKV("llama.vocab_size", valueTypeInt32, int32(32000))
	// int64 → mapUint64
	b.addKV("llama.block_count", valueTypeInt64, int64(32))
	// float64 → mapFloat32
	b.addKV("llama.rope.freq_base", valueTypeFloat64, float64(10000.0))
	// uint8 → mapUint32 (expert_count)
	b.addKV("llama.expert_count", valueTypeUint8, uint8(8))

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.Architecture.ContextLength != 8192 {
		t.Errorf("ContextLength = %d, want 8192", meta.Architecture.ContextLength)
	}
	if meta.Architecture.EmbeddingLength != 32 {
		t.Errorf("EmbeddingLength = %d, want 32", meta.Architecture.EmbeddingLength)
	}
	if meta.Architecture.VocabSize != 32000 {
		t.Errorf("VocabSize = %d, want 32000", meta.Architecture.VocabSize)
	}
	if meta.Architecture.BlockCount != 32 {
		t.Errorf("BlockCount = %d, want 32", meta.Architecture.BlockCount)
	}
	if meta.Architecture.ExpertCount != 8 {
		t.Errorf("ExpertCount = %d, want 8", meta.Architecture.ExpertCount)
	}
}

// TestReadMetadata_Uint64AsUint32 exercises uint64→uint32 coercion via mapUint32.
func TestReadMetadata_Uint64AsUint32(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "llama")
	b.addKV("general.quantization_version", valueTypeUint64, uint64(2))
	b.addKV("general.file_type", valueTypeUint64, uint64(15))

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if meta.General.QuantizationVersion != 2 {
		t.Errorf("QuantizationVersion = %d, want 2", meta.General.QuantizationVersion)
	}
	if meta.General.FileType != 15 {
		t.Errorf("FileType = %d, want 15", meta.General.FileType)
	}
}

// TestReadMetadata_Uint16AsUint32 exercises uint16→uint32 coercion via mapUint32.
func TestReadMetadata_Uint16AsUint32(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "llama")
	b.addKV("general.quantization_version", valueTypeUint16, uint16(3))
	b.addKV("llama.vocab_size", valueTypeUint16, uint16(128))

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if meta.General.QuantizationVersion != 3 {
		t.Errorf("QuantizationVersion = %d, want 3", meta.General.QuantizationVersion)
	}
	if meta.Architecture.VocabSize != 128 {
		t.Errorf("VocabSize = %d, want 128", meta.Architecture.VocabSize)
	}
}

// TestReadMetadata_Int64AsUint32 exercises int64→uint32 coercion.
func TestReadMetadata_Int64AsUint32(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "llama")
	b.addKV("llama.vocab_size", valueTypeInt64, int64(50257))

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if meta.Architecture.VocabSize != 50257 {
		t.Errorf("VocabSize = %d, want 50257", meta.Architecture.VocabSize)
	}
}

// TestReadMetadata_Uint32AsUint64 exercises uint32→uint64 coercion via mapUint64.
func TestReadMetadata_Uint32AsUint64(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "llama")
	b.addKV("llama.context_length", valueTypeUint32, uint32(4096))
	b.addKV("llama.rope.dimension_count", valueTypeUint32, uint32(64))

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if meta.Architecture.ContextLength != 4096 {
		t.Errorf("ContextLength = %d, want 4096", meta.Architecture.ContextLength)
	}
	if meta.Architecture.RoPEDimensionCount != 64 {
		t.Errorf("RoPEDimensionCount = %d, want 64", meta.Architecture.RoPEDimensionCount)
	}
}

// TestReadMetadata_Int32AsUint64 exercises int32→uint64 coercion.
func TestReadMetadata_Int32AsUint64(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "llama")
	b.addKV("llama.block_count", valueTypeInt32, int32(16))

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if meta.Architecture.BlockCount != 16 {
		t.Errorf("BlockCount = %d, want 16", meta.Architecture.BlockCount)
	}
}

// TestReadMetadata_Float64AsFloat32 exercises float64→float32 coercion.
func TestReadMetadata_Float64AsFloat32(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "llama")
	b.addKV("llama.rope.freq_base", valueTypeFloat64, float64(500000.0))
	b.addKV("llama.attention.layer_norm_rms_epsilon", valueTypeFloat64, float64(1e-6))

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if meta.Architecture.RoPEFreqBase <= 0 {
		t.Errorf("RoPEFreqBase = %v, want > 0", meta.Architecture.RoPEFreqBase)
	}
	if meta.Architecture.LayerNormRMSEpsilon <= 0 {
		t.Errorf("LayerNormRMSEpsilon = %v, want > 0", meta.Architecture.LayerNormRMSEpsilon)
	}
}

// TestSkipArray_FixedSizeElements exercises the fixed-size array skip path.
func TestSkipArray_FixedSizeElements(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "llama")

	// Add an array of uint32 elements (fixed size → one seek)
	// We need to build this manually since ggufBuilder only supports string arrays.
	// Instead, use a V3 builder with a numeric array to exercise skipArray.
	// Since the builder's writeValue only handles []string arrays, we build
	// a small custom test GGUF with a uint32 array.

	var buf bytes.Buffer
	writeU32le := func(v uint32) {
		b := make([]byte, 4)
		binary.LittleEndian.PutUint32(b, v)
		buf.Write(b)
	}
	writeU64le := func(v uint64) {
		b := make([]byte, 8)
		binary.LittleEndian.PutUint64(b, v)
		buf.Write(b)
	}
	writeStrV3 := func(s string) {
		writeU64le(uint64(len(s)))
		buf.WriteString(s)
	}

	// GGUF header
	writeU32le(0x46554747) // magic
	writeU32le(3)          // version
	writeU64le(0)          // tensor count
	writeU64le(2)          // 2 KV pairs

	// KV 1: uint32 array (fixed-size elements → fast skip)
	writeStrV3("skip.uint32_array")
	writeU32le(9) // valueTypeArray
	writeU32le(4) // elem type = uint32
	writeU64le(5) // 5 elements
	for i := 0; i < 5; i++ {
		writeU32le(uint32(i))
	}

	// KV 2: general.architecture
	writeStrV3("general.architecture")
	writeU32le(8) // valueTypeString
	writeStrV3("llama")

	meta, err := readMetadataFrom(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if meta.General.Architecture != "llama" {
		t.Errorf("architecture = %q, want %q", meta.General.Architecture, "llama")
	}
}

// TestReadMetadata_LongStringLimit verifies the string length sanity check.
func TestReadMetadata_LongStringLimit(t *testing.T) {
	var buf bytes.Buffer
	writeU32le := func(v uint32) {
		b := make([]byte, 4)
		binary.LittleEndian.PutUint32(b, v)
		buf.Write(b)
	}
	writeU64le := func(v uint64) {
		b := make([]byte, 8)
		binary.LittleEndian.PutUint64(b, v)
		buf.Write(b)
	}

	// GGUF header
	writeU32le(0x46554747)
	writeU32le(3)
	writeU64le(0)
	writeU64le(1)

	// Write a KV with a string value that has an impossibly large length
	// Key: "bad.key" (valid)
	writeU64le(7)
	buf.WriteString("bad.key")
	writeU32le(8)                // valueTypeString
	writeU64le(20 * 1024 * 1024) // 20MB string — exceeds 10MB sanity limit

	_, err := readMetadataFrom(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error for string exceeding sanity limit")
	}
}

// TestMapUint32_DefaultType verifies mapUint32 returns 0 for unrecognized types.
func TestMapUint32_DefaultType(t *testing.T) {
	m := map[string]any{"key": "not-a-number"}
	if v := mapUint32(m, "key"); v != 0 {
		t.Errorf("mapUint32 with string value = %d, want 0", v)
	}
}

// TestMapUint64_DefaultType verifies mapUint64 returns 0 for unrecognized types.
func TestMapUint64_DefaultType(t *testing.T) {
	m := map[string]any{"key": "not-a-number"}
	if v := mapUint64(m, "key"); v != 0 {
		t.Errorf("mapUint64 with string value = %d, want 0", v)
	}
}

// TestMapFloat32_DefaultType verifies mapFloat32 returns 0 for unrecognized types.
func TestMapFloat32_DefaultType(t *testing.T) {
	m := map[string]any{"key": "not-a-float"}
	if v := mapFloat32(m, "key"); v != 0 {
		t.Errorf("mapFloat32 with string value = %v, want 0", v)
	}
}

// TestReadString_V2_Length verifies V2 uses uint32 for string length.
func TestReadString_V2_Length(t *testing.T) {
	b := newBuilder(2) // V2 uses uint32 for string lengths
	b.addKV("general.architecture", valueTypeString, "llama")
	b.addKV("general.name", valueTypeString, "LlamaV2")

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if meta.General.Name != "LlamaV2" {
		t.Errorf("name = %q, want %q", meta.General.Name, "LlamaV2")
	}
}
