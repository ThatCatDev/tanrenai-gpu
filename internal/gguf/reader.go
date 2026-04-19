package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// GGUF magic number: "GGUF" in little-endian.
const ggufMagic = 0x46554747 // 'G','G','U','F'

// GGUF value types.
const (
	valueTypeUint8   = 0
	valueTypeInt8    = 1
	valueTypeUint16  = 2
	valueTypeInt16   = 3
	valueTypeUint32  = 4
	valueTypeInt32   = 5
	valueTypeFloat32 = 6
	valueTypeBool    = 7
	valueTypeString  = 8
	valueTypeArray   = 9
	valueTypeUint64  = 10
	valueTypeInt64   = 11
	valueTypeFloat64 = 12
)

// Fixed byte sizes for non-variable value types.
var valueFixedSize = map[uint32]int64{
	valueTypeUint8:   1,
	valueTypeInt8:    1,
	valueTypeUint16:  2,
	valueTypeInt16:   2,
	valueTypeUint32:  4,
	valueTypeInt32:   4,
	valueTypeFloat32: 4,
	valueTypeBool:    1,
	valueTypeUint64:  8,
	valueTypeInt64:   8,
	valueTypeFloat64: 8,
}

// Metadata holds fields extracted from a GGUF file header.
type Metadata struct {
	Version      uint32
	General      General
	Architecture Architecture // arch-specific params (keys prefixed with general.architecture)
	Tokenizer    Tokenizer
}

// General holds values from the general.* namespace.
type General struct {
	Architecture        string // general.architecture (e.g. "qwen2", "llama")
	Name                string // general.name
	QuantizationVersion uint32 // general.quantization_version
	FileType            uint32 // general.file_type
	SizeLabel           string // general.size_label (e.g. "7B", "34B")
}

// Architecture holds values from the {arch}.* namespace.
type Architecture struct {
	ContextLength       uint64  // {arch}.context_length
	EmbeddingLength     uint64  // {arch}.embedding_length
	BlockCount          uint64  // {arch}.block_count
	FeedForwardLength   uint64  // {arch}.feed_forward_length
	HeadCount           uint64  // {arch}.attention.head_count
	HeadCountKV         uint64  // {arch}.attention.head_count_kv
	LayerNormRMSEpsilon float32 // {arch}.attention.layer_norm_rms_epsilon
	VocabSize           uint32  // {arch}.vocab_size
	RoPEDimensionCount  uint64  // {arch}.rope.dimension_count
	RoPEFreqBase        float32 // {arch}.rope.freq_base
	RoPEScalingType     string  // {arch}.rope.scaling.type
	RoPEScalingFactor   float32 // {arch}.rope.scaling.factor
	RoPEOrigCtxLength   uint32  // {arch}.rope.scaling.original_context_length
	ExpertCount         uint32  // {arch}.expert_count
	ExpertUsedCount     uint32  // {arch}.expert_used_count
}

// Tokenizer holds values from the tokenizer.* namespace.
type Tokenizer struct {
	ChatTemplate   string // tokenizer.chat_template
	Model          string // tokenizer.ggml.model (e.g. "llama", "gpt2", "bpe")
	BOSTokenID     uint32 // tokenizer.ggml.bos_token_id
	EOSTokenID     uint32 // tokenizer.ggml.eos_token_id
	UnknownTokenID uint32 // tokenizer.ggml.unknown_token_id
	PaddingTokenID uint32 // tokenizer.ggml.padding_token_id
	AddBOSToken    bool   // tokenizer.ggml.add_bos_token
	AddEOSToken    bool   // tokenizer.ggml.add_eos_token
}

// ReadMetadata reads only the GGUF header and metadata KV pairs from the given
// file. It does not read tensor data. Supports GGUF v2 and v3.
func ReadMetadata(path string) (*Metadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open: %w", err)
	}
	defer func() { _ = f.Close() }()

	return readMetadataFrom(f)
}

func readMetadataFrom(r io.ReadSeeker) (*Metadata, error) {
	// Read magic.
	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("gguf: invalid magic 0x%08X (expected 0x%08X)", magic, ggufMagic)
	}

	// Read version.
	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("gguf: read version: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", version)
	}

	// Read tensor count and KV count.
	// v2 uses uint32, v3 uses uint64.
	var tensorCount, kvCount uint64
	var err error
	if version == 2 {
		tensorCount, kvCount, err = readCountsV2(r)
	} else {
		tensorCount, kvCount, err = readCountsV3(r)
	}
	if err != nil {
		return nil, err
	}
	_ = tensorCount // we only need KV pairs

	// Pass 1: read all KV pairs into a map.
	kvMap := make(map[string]any, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key, err := readString(r, version)
		if err != nil {
			return nil, fmt.Errorf("gguf: kv %d key: %w", i, err)
		}

		var valueType uint32
		if err := binary.Read(r, binary.LittleEndian, &valueType); err != nil {
			return nil, fmt.Errorf("gguf: kv %d value type: %w", i, err)
		}

		val, err := readValue(r, version, valueType)
		if err != nil {
			return nil, fmt.Errorf("gguf: kv %d value: %w", i, err)
		}
		if val != nil {
			kvMap[key] = val
		}
	}

	// Pass 2: populate typed struct from map.
	meta := &Metadata{Version: version}

	// General namespace.
	meta.General.Architecture = mapString(kvMap, "general.architecture")
	meta.General.Name = mapString(kvMap, "general.name")
	meta.General.QuantizationVersion = mapUint32(kvMap, "general.quantization_version")
	meta.General.FileType = mapUint32(kvMap, "general.file_type")
	meta.General.SizeLabel = mapString(kvMap, "general.size_label")

	// Architecture namespace (keyed by general.architecture).
	arch := meta.General.Architecture
	if arch != "" {
		meta.Architecture.ContextLength = mapUint64(kvMap, arch+".context_length")
		meta.Architecture.EmbeddingLength = mapUint64(kvMap, arch+".embedding_length")
		meta.Architecture.BlockCount = mapUint64(kvMap, arch+".block_count")
		meta.Architecture.FeedForwardLength = mapUint64(kvMap, arch+".feed_forward_length")
		meta.Architecture.HeadCount = mapUint64(kvMap, arch+".attention.head_count")
		meta.Architecture.HeadCountKV = mapUint64(kvMap, arch+".attention.head_count_kv")
		meta.Architecture.LayerNormRMSEpsilon = mapFloat32(kvMap, arch+".attention.layer_norm_rms_epsilon")
		meta.Architecture.VocabSize = mapUint32(kvMap, arch+".vocab_size")
		meta.Architecture.RoPEDimensionCount = mapUint64(kvMap, arch+".rope.dimension_count")
		meta.Architecture.RoPEFreqBase = mapFloat32(kvMap, arch+".rope.freq_base")
		meta.Architecture.RoPEScalingType = mapString(kvMap, arch+".rope.scaling.type")
		meta.Architecture.RoPEScalingFactor = mapFloat32(kvMap, arch+".rope.scaling.factor")
		meta.Architecture.RoPEOrigCtxLength = mapUint32(kvMap, arch+".rope.scaling.original_context_length")
		meta.Architecture.ExpertCount = mapUint32(kvMap, arch+".expert_count")
		meta.Architecture.ExpertUsedCount = mapUint32(kvMap, arch+".expert_used_count")
	}

	// Tokenizer namespace.
	meta.Tokenizer.ChatTemplate = mapString(kvMap, "tokenizer.chat_template")
	meta.Tokenizer.Model = mapString(kvMap, "tokenizer.ggml.model")
	meta.Tokenizer.BOSTokenID = mapUint32(kvMap, "tokenizer.ggml.bos_token_id")
	meta.Tokenizer.EOSTokenID = mapUint32(kvMap, "tokenizer.ggml.eos_token_id")
	meta.Tokenizer.UnknownTokenID = mapUint32(kvMap, "tokenizer.ggml.unknown_token_id")
	meta.Tokenizer.PaddingTokenID = mapUint32(kvMap, "tokenizer.ggml.padding_token_id")
	meta.Tokenizer.AddBOSToken = mapBool(kvMap, "tokenizer.ggml.add_bos_token")
	meta.Tokenizer.AddEOSToken = mapBool(kvMap, "tokenizer.ggml.add_eos_token")

	return meta, nil
}

// readCountsV2 reads the tensor count and KV count fields for GGUF v2 (uint32 each).
func readCountsV2(r io.Reader) (tensorCount, kvCount uint64, err error) {
	var tc, kc uint32
	if err = binary.Read(r, binary.LittleEndian, &tc); err != nil {
		return 0, 0, fmt.Errorf("gguf: read tensor count: %w", err)
	}
	if err = binary.Read(r, binary.LittleEndian, &kc); err != nil {
		return 0, 0, fmt.Errorf("gguf: read kv count: %w", err)
	}

	return uint64(tc), uint64(kc), nil
}

// readCountsV3 reads the tensor count and KV count fields for GGUF v3 (uint64 each).
func readCountsV3(r io.Reader) (tensorCount, kvCount uint64, err error) {
	if err = binary.Read(r, binary.LittleEndian, &tensorCount); err != nil {
		return 0, 0, fmt.Errorf("gguf: read tensor count: %w", err)
	}
	if err = binary.Read(r, binary.LittleEndian, &kvCount); err != nil {
		return 0, 0, fmt.Errorf("gguf: read kv count: %w", err)
	}

	return tensorCount, kvCount, nil
}

// readString reads a GGUF string (length-prefixed).
// v2 uses uint32 lengths, v3 uses uint64.
func readString(r io.Reader, version uint32) (string, error) {
	var length uint64
	if version == 2 {
		var l uint32
		if err := binary.Read(r, binary.LittleEndian, &l); err != nil {
			return "", err
		}
		length = uint64(l)
	} else {
		if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
			return "", err
		}
	}

	if length > 10*1024*1024 { // 10 MB sanity limit
		return "", fmt.Errorf("string length %d exceeds sanity limit", length)
	}

	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}

	return string(buf), nil
}

// readValue reads a value of the given type. Returns nil for array values
// (large arrays like tokenizer vocab are skipped to save memory).
func readValue(r io.ReadSeeker, version, valueType uint32) (any, error) {
	switch valueType {
	case valueTypeUint8:
		var v uint8

		return v, binary.Read(r, binary.LittleEndian, &v)
	case valueTypeInt8:
		var v int8

		return v, binary.Read(r, binary.LittleEndian, &v)
	case valueTypeUint16:
		var v uint16

		return v, binary.Read(r, binary.LittleEndian, &v)
	case valueTypeInt16:
		var v int16

		return v, binary.Read(r, binary.LittleEndian, &v)
	case valueTypeUint32:
		var v uint32

		return v, binary.Read(r, binary.LittleEndian, &v)
	case valueTypeInt32:
		var v int32

		return v, binary.Read(r, binary.LittleEndian, &v)
	case valueTypeFloat32:
		var v float32

		return v, binary.Read(r, binary.LittleEndian, &v)
	case valueTypeBool:
		var v uint8
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return false, err
		}

		return v != 0, nil
	case valueTypeString:
		return readString(r, version)
	case valueTypeUint64:
		var v uint64

		return v, binary.Read(r, binary.LittleEndian, &v)
	case valueTypeInt64:
		var v int64

		return v, binary.Read(r, binary.LittleEndian, &v)
	case valueTypeFloat64:
		var v float64

		return v, binary.Read(r, binary.LittleEndian, &v)
	case valueTypeArray:
		// Skip arrays — they can be very large (tokenizer vocab/merges/scores).
		if err := skipArray(r, version); err != nil {
			return nil, err
		}

		return nil, nil
	default:
		return nil, fmt.Errorf("unknown value type %d", valueType)
	}
}

// skipArray seeks past an array value without loading it into memory.
func skipArray(r io.ReadSeeker, version uint32) error {
	var elemType uint32
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return err
	}
	var count uint64
	if version == 2 {
		var c uint32
		if err := binary.Read(r, binary.LittleEndian, &c); err != nil {
			return err
		}
		count = uint64(c)
	} else {
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return err
		}
	}

	// If elements have fixed size, skip in one seek.
	if sz, ok := valueFixedSize[elemType]; ok {
		_, err := r.Seek(int64(count)*sz, io.SeekCurrent)

		return err
	}

	// Otherwise skip each element individually.
	for j := uint64(0); j < count; j++ {
		if _, err := readValue(r, version, elemType); err != nil {
			return err
		}
	}

	return nil
}

// Type-safe extraction helpers. They coerce numeric types to the target type.

func mapString(m map[string]any, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}

	return ""
}

func mapUint32(m map[string]any, key string) uint32 {
	v, ok := m[key]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case uint32:
		return n
	case uint64:
		return uint32(n)
	case uint16:
		return uint32(n)
	case uint8:
		return uint32(n)
	case int32:
		return uint32(n)
	case int64:
		return uint32(n)
	default:
		return 0
	}
}

func mapUint64(m map[string]any, key string) uint64 {
	v, ok := m[key]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case uint64:
		return n
	case uint32:
		return uint64(n)
	case uint16:
		return uint64(n)
	case uint8:
		return uint64(n)
	case int64:
		return uint64(n)
	case int32:
		return uint64(n)
	default:
		return 0
	}
}

func mapFloat32(m map[string]any, key string) float32 {
	v, ok := m[key]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case float32:
		return n
	case float64:
		return float32(n)
	default:
		return 0
	}
}

func mapBool(m map[string]any, key string) bool {
	if v, ok := m[key]; ok {
		if b, ok := v.(bool); ok {
			return b
		}
	}

	return false
}
