"""FastAPI sidecar for fine-tuning via Unsloth."""

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from train import run_training

app = FastAPI(title="Tanrenai Training Sidecar")

# Global state: only one training job at a time.
_current_run: Optional[str] = None
_lock = threading.Lock()
_runs: dict[str, dict] = {}


class TrainRequest(BaseModel):
    dataset_path: str
    base_model_path: str
    output_dir: str
    run_id: Optional[str] = None
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    batch_size: int = 4


class MergeRequest(BaseModel):
    base_model_path: str
    adapter_dir: str
    output_path: str


class ConvertRequest(BaseModel):
    model_dir: str
    output_path: str
    quantization: str = "Q4_K_M"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/train")
def start_training(req: TrainRequest):
    global _current_run

    with _lock:
        if _current_run is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Training already in progress: {_current_run}",
            )
        run_id = req.run_id or str(uuid.uuid4())[:12]
        _current_run = run_id
        _runs[run_id] = {"status": "training", "metrics": {}}

    def _train():
        global _current_run
        try:
            metrics = run_training(
                dataset_path=req.dataset_path,
                base_model=req.base_model_path,
                output_dir=req.output_dir,
                epochs=req.epochs,
                learning_rate=req.learning_rate,
                lora_rank=req.lora_rank,
                lora_alpha=req.lora_alpha,
                batch_size=req.batch_size,
            )
            _runs[run_id] = {"status": "done", "metrics": metrics}
        except Exception as e:
            _runs[run_id] = {"status": "failed", "error": str(e)}
        finally:
            with _lock:
                _current_run = None

    thread = threading.Thread(target=_train, daemon=True)
    thread.start()

    return {"run_id": run_id, "status": "training"}


@app.get("/status/{run_id}")
def get_status(run_id: str):
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    info = _runs[run_id]

    # Try to read live metrics from the output dir
    # The training callback writes metrics.json during training
    if info["status"] == "training":
        for run_data in _runs.values():
            pass  # metrics are updated in-place by the callback

    return info


@app.post("/merge")
def merge_adapter(req: MergeRequest):
    """Merge a LoRA adapter back into the base model."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=req.base_model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )

        # Load and merge the adapter
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, req.adapter_dir)
        model = model.merge_and_unload()

        os.makedirs(req.output_path, exist_ok=True)
        model.save_pretrained(req.output_path)
        tokenizer.save_pretrained(req.output_path)

        return {"status": "ok", "output_path": req.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/convert")
def convert_to_gguf(req: ConvertRequest):
    """Convert a merged model to GGUF format using llama.cpp's convert script."""
    import subprocess

    try:
        # Try to find convert_hf_to_gguf.py in common locations
        convert_script = None
        search_paths = [
            Path.home() / ".local" / "share" / "tanrenai" / "bin" / "convert_hf_to_gguf.py",
            Path("/usr/local/bin/convert_hf_to_gguf.py"),
            Path("convert_hf_to_gguf.py"),
        ]
        for p in search_paths:
            if p.exists():
                convert_script = str(p)
                break

        if convert_script is None:
            raise FileNotFoundError(
                "convert_hf_to_gguf.py not found. "
                "Place it in ~/.local/share/tanrenai/bin/"
            )

        os.makedirs(os.path.dirname(req.output_path), exist_ok=True)

        cmd = [
            "python3", convert_script,
            req.model_dir,
            "--outfile", req.output_path,
            "--outtype", req.quantization.lower(),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed: {result.stderr}")

        return {"status": "ok", "output_path": req.output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
