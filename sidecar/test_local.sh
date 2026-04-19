#!/bin/bash
# Local test script for the fine-tuning pipeline.
# Prereqs: run setup.sh first
#
# Usage:
#   cd server/sidecar
#   bash setup.sh       # one-time setup
#   bash test_local.sh  # run the test

set -e

SIDECAR_PORT=18082
BASE_MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${HOME}/.local/share/tanrenai/training"

echo "=== Phase 5 Local Test ==="
echo "Model: ${BASE_MODEL}"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Activate venv
if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    echo "ERROR: venv not found. Run setup.sh first."
    exit 1
fi
source "${SCRIPT_DIR}/venv/bin/activate"

# Create test dataset
echo "--- Creating test dataset ---"
mkdir -p "${DATA_DIR}/datasets"
TEST_DATASET="${DATA_DIR}/datasets/test-run.jsonl"
cat > "${TEST_DATASET}" << 'JSONL'
{"messages":[{"role":"system","content":"You are a helpful coding assistant."},{"role":"user","content":"Write a Python function to check if a number is prime."},{"role":"assistant","content":"```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```"}]}
{"messages":[{"role":"system","content":"You are a helpful coding assistant."},{"role":"user","content":"How do I read a file in Go?"},{"role":"assistant","content":"Use os.ReadFile for the simplest approach:\n\n```go\ndata, err := os.ReadFile(\"path/to/file\")\nif err != nil {\n    log.Fatal(err)\n}\nfmt.Println(string(data))\n```"}]}
{"messages":[{"role":"system","content":"You are a helpful coding assistant."},{"role":"user","content":"Explain what a goroutine is."},{"role":"assistant","content":"A goroutine is a lightweight thread of execution managed by the Go runtime. You start one with the `go` keyword before a function call. Goroutines are multiplexed onto OS threads and are very cheap to create — you can have thousands running simultaneously."}]}
{"messages":[{"role":"system","content":"You are a helpful coding assistant."},{"role":"user","content":"Write a bash one-liner to find all .go files larger than 1MB."},{"role":"assistant","content":"```bash\nfind . -name '*.go' -size +1M -ls\n```"}]}
{"messages":[{"role":"system","content":"You are a helpful coding assistant."},{"role":"user","content":"What's the difference between a slice and an array in Go?"},{"role":"assistant","content":"An array has a fixed size that's part of its type (e.g. [5]int), while a slice is a dynamically-sized view into an array. Slices are what you use 99% of the time — they have a length, capacity, and a pointer to the underlying array. You create them with make() or slice literals."}]}
JSONL
echo "Created ${TEST_DATASET} with 5 samples"

# Start sidecar
echo "--- Starting sidecar on port ${SIDECAR_PORT} ---"
uvicorn main:app --host 127.0.0.1 --port ${SIDECAR_PORT} &
SIDECAR_PID=$!
sleep 2

# Health check
if curl -sf http://127.0.0.1:${SIDECAR_PORT}/health > /dev/null; then
    echo "Sidecar is healthy"
else
    echo "ERROR: Sidecar failed to start"
    kill ${SIDECAR_PID} 2>/dev/null
    exit 1
fi

# Start training
echo ""
echo "--- Starting training ---"
mkdir -p "${DATA_DIR}/runs/test-run"
TRAIN_RESP=$(curl -sf -X POST http://127.0.0.1:${SIDECAR_PORT}/train \
    -H "Content-Type: application/json" \
    -d "{
        \"dataset_path\": \"${TEST_DATASET}\",
        \"base_model_path\": \"${BASE_MODEL}\",
        \"output_dir\": \"${DATA_DIR}/runs/test-run\",
        \"run_id\": \"test-run\",
        \"epochs\": 1,
        \"learning_rate\": 2e-4,
        \"lora_rank\": 8,
        \"lora_alpha\": 16,
        \"batch_size\": 2
    }")
echo "Response: ${TRAIN_RESP}"

# Poll status
echo ""
echo "--- Polling training status ---"
for i in $(seq 1 120); do
    STATUS=$(curl -sf http://127.0.0.1:${SIDECAR_PORT}/status/test-run)
    STAT=$(echo "${STATUS}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null)
    echo "[${i}] status=${STAT}"

    if [ "${STAT}" = "done" ]; then
        echo ""
        echo "=== Training complete! ==="
        echo "${STATUS}" | python3 -m json.tool
        break
    elif [ "${STAT}" = "failed" ]; then
        echo ""
        echo "=== Training failed ==="
        echo "${STATUS}" | python3 -m json.tool
        break
    fi

    sleep 5
done

# Cleanup
echo ""
echo "--- Stopping sidecar ---"
kill ${SIDECAR_PID} 2>/dev/null
wait ${SIDECAR_PID} 2>/dev/null
echo "Done."
