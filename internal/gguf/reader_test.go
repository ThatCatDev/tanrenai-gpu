package gguf

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"testing"
)

// ggufBuilder helps construct minimal GGUF files for testing.
type ggufBuilder struct {
	buf     bytes.Buffer
	version uint32
	kvs     []kv
}

type kv struct {
	key       string
	valueType uint32
	value     interface{} // string, uint32, int32, float32, bool, []string, etc.
}

func newBuilder(version uint32) *ggufBuilder {
	return &ggufBuilder{version: version}
}

func (b *ggufBuilder) addKV(key string, valueType uint32, value interface{}) {
	b.kvs = append(b.kvs, kv{key: key, valueType: valueType, value: value})
}

func (b *ggufBuilder) build() *bytes.Reader {
	b.buf.Reset()

	// Magic
	binary.Write(&b.buf, binary.LittleEndian, uint32(ggufMagic))
	// Version
	binary.Write(&b.buf, binary.LittleEndian, b.version)
	// Tensor count, KV count
	if b.version == 2 {
		binary.Write(&b.buf, binary.LittleEndian, uint32(0)) // tensors
		binary.Write(&b.buf, binary.LittleEndian, uint32(len(b.kvs)))
	} else {
		binary.Write(&b.buf, binary.LittleEndian, uint64(0)) // tensors
		binary.Write(&b.buf, binary.LittleEndian, uint64(len(b.kvs)))
	}

	for _, kv := range b.kvs {
		b.writeString(kv.key)
		binary.Write(&b.buf, binary.LittleEndian, kv.valueType)
		b.writeValue(kv.valueType, kv.value)
	}

	return bytes.NewReader(b.buf.Bytes())
}

func (b *ggufBuilder) writeString(s string) {
	if b.version == 2 {
		binary.Write(&b.buf, binary.LittleEndian, uint32(len(s)))
	} else {
		binary.Write(&b.buf, binary.LittleEndian, uint64(len(s)))
	}
	b.buf.WriteString(s)
}

func (b *ggufBuilder) writeValue(valueType uint32, value interface{}) {
	switch valueType {
	case valueTypeString:
		b.writeString(value.(string))
	case valueTypeUint8:
		binary.Write(&b.buf, binary.LittleEndian, value.(uint8))
	case valueTypeInt8:
		binary.Write(&b.buf, binary.LittleEndian, value.(int8))
	case valueTypeUint16:
		binary.Write(&b.buf, binary.LittleEndian, value.(uint16))
	case valueTypeInt16:
		binary.Write(&b.buf, binary.LittleEndian, value.(int16))
	case valueTypeUint32:
		binary.Write(&b.buf, binary.LittleEndian, value.(uint32))
	case valueTypeInt32:
		binary.Write(&b.buf, binary.LittleEndian, value.(int32))
	case valueTypeFloat32:
		binary.Write(&b.buf, binary.LittleEndian, value.(float32))
	case valueTypeBool:
		if value.(bool) {
			b.buf.WriteByte(1)
		} else {
			b.buf.WriteByte(0)
		}
	case valueTypeUint64:
		binary.Write(&b.buf, binary.LittleEndian, value.(uint64))
	case valueTypeInt64:
		binary.Write(&b.buf, binary.LittleEndian, value.(int64))
	case valueTypeFloat64:
		binary.Write(&b.buf, binary.LittleEndian, value.(float64))
	case valueTypeArray:
		arr := value.([]string)
		binary.Write(&b.buf, binary.LittleEndian, uint32(valueTypeString)) // elem type
		if b.version == 2 {
			binary.Write(&b.buf, binary.LittleEndian, uint32(len(arr)))
		} else {
			binary.Write(&b.buf, binary.LittleEndian, uint64(len(arr)))
		}
		for _, s := range arr {
			b.writeString(s)
		}
	}
}

func TestReadMetadata_V3_AllKeys(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "qwen2")
	b.addKV("general.name", valueTypeString, "Qwen2.5-Coder-32B-Instruct")
	b.addKV("tokenizer.chat_template", valueTypeString, "{{ messages }}")

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.Version != 3 {
		t.Errorf("version = %d, want 3", meta.Version)
	}
	if meta.General.Architecture != "qwen2" {
		t.Errorf("architecture = %q, want %q", meta.General.Architecture, "qwen2")
	}
	if meta.General.Name != "Qwen2.5-Coder-32B-Instruct" {
		t.Errorf("name = %q, want %q", meta.General.Name, "Qwen2.5-Coder-32B-Instruct")
	}
	if meta.Tokenizer.ChatTemplate != "{{ messages }}" {
		t.Errorf("chat_template = %q, want %q", meta.Tokenizer.ChatTemplate, "{{ messages }}")
	}
}

func TestReadMetadata_V2(t *testing.T) {
	b := newBuilder(2)
	b.addKV("general.architecture", valueTypeString, "llama")
	b.addKV("general.name", valueTypeString, "Llama-3")

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.Version != 2 {
		t.Errorf("version = %d, want 2", meta.Version)
	}
	if meta.General.Architecture != "llama" {
		t.Errorf("architecture = %q, want %q", meta.General.Architecture, "llama")
	}
	if meta.General.Name != "Llama-3" {
		t.Errorf("name = %q, want %q", meta.General.Name, "Llama-3")
	}
}

func TestReadMetadata_ArchFields(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "qwen2")
	b.addKV("qwen2.context_length", valueTypeUint32, uint32(32768))
	b.addKV("qwen2.embedding_length", valueTypeUint32, uint32(4096))
	b.addKV("qwen2.block_count", valueTypeUint64, uint64(64))
	b.addKV("qwen2.feed_forward_length", valueTypeUint32, uint32(11008))
	b.addKV("qwen2.attention.head_count", valueTypeUint32, uint32(32))
	b.addKV("qwen2.attention.head_count_kv", valueTypeUint32, uint32(8))
	b.addKV("qwen2.vocab_size", valueTypeUint32, uint32(151936))
	b.addKV("qwen2.expert_count", valueTypeUint32, uint32(8))
	b.addKV("qwen2.expert_used_count", valueTypeUint32, uint32(2))

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.Architecture.ContextLength != 32768 {
		t.Errorf("context_length = %d, want 32768", meta.Architecture.ContextLength)
	}
	if meta.Architecture.EmbeddingLength != 4096 {
		t.Errorf("embedding_length = %d, want 4096", meta.Architecture.EmbeddingLength)
	}
	if meta.Architecture.BlockCount != 64 {
		t.Errorf("block_count = %d, want 64", meta.Architecture.BlockCount)
	}
	if meta.Architecture.FeedForwardLength != 11008 {
		t.Errorf("feed_forward_length = %d, want 11008", meta.Architecture.FeedForwardLength)
	}
	if meta.Architecture.HeadCount != 32 {
		t.Errorf("head_count = %d, want 32", meta.Architecture.HeadCount)
	}
	if meta.Architecture.HeadCountKV != 8 {
		t.Errorf("head_count_kv = %d, want 8", meta.Architecture.HeadCountKV)
	}
	if meta.Architecture.VocabSize != 151936 {
		t.Errorf("vocab_size = %d, want 151936", meta.Architecture.VocabSize)
	}
	if meta.Architecture.ExpertCount != 8 {
		t.Errorf("expert_count = %d, want 8", meta.Architecture.ExpertCount)
	}
	if meta.Architecture.ExpertUsedCount != 2 {
		t.Errorf("expert_used_count = %d, want 2", meta.Architecture.ExpertUsedCount)
	}
}

func TestReadMetadata_RoPEFloatFields(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "llama")
	b.addKV("llama.attention.layer_norm_rms_epsilon", valueTypeFloat32, float32(1e-5))
	b.addKV("llama.rope.freq_base", valueTypeFloat32, float32(10000.0))
	b.addKV("llama.rope.dimension_count", valueTypeUint64, uint64(128))
	b.addKV("llama.rope.scaling.type", valueTypeString, "linear")
	b.addKV("llama.rope.scaling.factor", valueTypeFloat32, float32(2.0))
	b.addKV("llama.rope.scaling.original_context_length", valueTypeUint32, uint32(4096))

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if math.Float32bits(meta.Architecture.LayerNormRMSEpsilon) != math.Float32bits(1e-5) {
		t.Errorf("layer_norm_rms_epsilon = %v, want 1e-5", meta.Architecture.LayerNormRMSEpsilon)
	}
	if math.Float32bits(meta.Architecture.RoPEFreqBase) != math.Float32bits(10000.0) {
		t.Errorf("rope_freq_base = %v, want 10000.0", meta.Architecture.RoPEFreqBase)
	}
	if meta.Architecture.RoPEDimensionCount != 128 {
		t.Errorf("rope_dimension_count = %d, want 128", meta.Architecture.RoPEDimensionCount)
	}
	if meta.Architecture.RoPEScalingType != "linear" {
		t.Errorf("rope_scaling_type = %q, want %q", meta.Architecture.RoPEScalingType, "linear")
	}
	if math.Float32bits(meta.Architecture.RoPEScalingFactor) != math.Float32bits(2.0) {
		t.Errorf("rope_scaling_factor = %v, want 2.0", meta.Architecture.RoPEScalingFactor)
	}
	if meta.Architecture.RoPEOrigCtxLength != 4096 {
		t.Errorf("rope_orig_ctx_length = %d, want 4096", meta.Architecture.RoPEOrigCtxLength)
	}
}

func TestReadMetadata_NoArchitecture(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.name", valueTypeString, "TestModel")
	b.addKV("tokenizer.chat_template", valueTypeString, "{{ messages }}")

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.General.Architecture != "" {
		t.Errorf("architecture = %q, want empty", meta.General.Architecture)
	}
	// All arch fields should be zero-valued.
	if meta.Architecture.ContextLength != 0 {
		t.Errorf("context_length = %d, want 0", meta.Architecture.ContextLength)
	}
	if meta.Architecture.HeadCount != 0 {
		t.Errorf("head_count = %d, want 0", meta.Architecture.HeadCount)
	}
}

func TestReadMetadata_GeneralFields(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "llama")
	b.addKV("general.name", valueTypeString, "Llama-3-70B")
	b.addKV("general.quantization_version", valueTypeUint32, uint32(2))
	b.addKV("general.file_type", valueTypeUint32, uint32(7))
	b.addKV("general.size_label", valueTypeString, "70B")

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.General.QuantizationVersion != 2 {
		t.Errorf("quantization_version = %d, want 2", meta.General.QuantizationVersion)
	}
	if meta.General.FileType != 7 {
		t.Errorf("file_type = %d, want 7", meta.General.FileType)
	}
	if meta.General.SizeLabel != "70B" {
		t.Errorf("size_label = %q, want %q", meta.General.SizeLabel, "70B")
	}
}

func TestReadMetadata_TokenizerFields(t *testing.T) {
	b := newBuilder(3)
	b.addKV("tokenizer.chat_template", valueTypeString, "{{ messages }}")
	b.addKV("tokenizer.ggml.model", valueTypeString, "llama")
	b.addKV("tokenizer.ggml.bos_token_id", valueTypeUint32, uint32(1))
	b.addKV("tokenizer.ggml.eos_token_id", valueTypeUint32, uint32(2))
	b.addKV("tokenizer.ggml.unknown_token_id", valueTypeUint32, uint32(0))
	b.addKV("tokenizer.ggml.padding_token_id", valueTypeUint32, uint32(3))
	b.addKV("tokenizer.ggml.add_bos_token", valueTypeBool, true)
	b.addKV("tokenizer.ggml.add_eos_token", valueTypeBool, false)

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.Tokenizer.Model != "llama" {
		t.Errorf("tokenizer model = %q, want %q", meta.Tokenizer.Model, "llama")
	}
	if meta.Tokenizer.BOSTokenID != 1 {
		t.Errorf("bos_token_id = %d, want 1", meta.Tokenizer.BOSTokenID)
	}
	if meta.Tokenizer.EOSTokenID != 2 {
		t.Errorf("eos_token_id = %d, want 2", meta.Tokenizer.EOSTokenID)
	}
	if meta.Tokenizer.UnknownTokenID != 0 {
		t.Errorf("unknown_token_id = %d, want 0", meta.Tokenizer.UnknownTokenID)
	}
	if meta.Tokenizer.PaddingTokenID != 3 {
		t.Errorf("padding_token_id = %d, want 3", meta.Tokenizer.PaddingTokenID)
	}
	if !meta.Tokenizer.AddBOSToken {
		t.Error("add_bos_token = false, want true")
	}
	if meta.Tokenizer.AddEOSToken {
		t.Error("add_eos_token = true, want false")
	}
}

func TestReadMetadata_SkipAllValueTypes(t *testing.T) {
	// Ensure we can skip every fixed-size value type, plus arrays.
	b := newBuilder(3)
	b.addKV("skip.uint8", valueTypeUint8, uint8(1))
	b.addKV("skip.int8", valueTypeInt8, int8(-1))
	b.addKV("skip.uint16", valueTypeUint16, uint16(2))
	b.addKV("skip.int16", valueTypeInt16, int16(-2))
	b.addKV("skip.uint32", valueTypeUint32, uint32(3))
	b.addKV("skip.int32", valueTypeInt32, int32(-3))
	b.addKV("skip.float32", valueTypeFloat32, float32(1.5))
	b.addKV("skip.bool", valueTypeBool, true)
	b.addKV("skip.uint64", valueTypeUint64, uint64(4))
	b.addKV("skip.int64", valueTypeInt64, int64(-4))
	b.addKV("skip.float64", valueTypeFloat64, float64(2.5))
	b.addKV("skip.string", valueTypeString, "hello")
	b.addKV("skip.array", valueTypeArray, []string{"a", "b"})
	b.addKV("general.architecture", valueTypeString, "found")

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.General.Architecture != "found" {
		t.Errorf("architecture = %q, want %q", meta.General.Architecture, "found")
	}
}

func TestReadMetadata_InvalidMagic(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(0xDEADBEEF))

	_, err := readMetadataFrom(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error for invalid magic")
	}
}

func TestReadMetadata_UnsupportedVersion(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(ggufMagic))
	binary.Write(&buf, binary.LittleEndian, uint32(99))

	_, err := readMetadataFrom(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error for unsupported version")
	}
}

func TestReadMetadata_TruncatedFile(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(ggufMagic))
	// No version — truncated

	_, err := readMetadataFrom(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error for truncated file")
	}
}

func TestReadMetadata_EmptyFile(t *testing.T) {
	_, err := readMetadataFrom(bytes.NewReader(nil))
	if err == nil {
		t.Fatal("expected error for empty file")
	}
}

func TestReadMetadata_NoTargetKeys(t *testing.T) {
	b := newBuilder(3)
	b.addKV("other.key", valueTypeString, "value")

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.General.Architecture != "" || meta.General.Name != "" || meta.Tokenizer.ChatTemplate != "" {
		t.Errorf("expected all empty, got arch=%q name=%q tpl=%q",
			meta.General.Architecture, meta.General.Name, meta.Tokenizer.ChatTemplate)
	}
}

// readMetadataFrom wraps the internal reader for testing with bytes.Reader.
// We need to make the function work with io.ReadSeeker.
func init() {
	// Verify bytes.Reader implements io.ReadSeeker.
	var _ io.ReadSeeker = (*bytes.Reader)(nil)
}
