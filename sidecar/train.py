"""Training logic using Unsloth for LoRA fine-tuning."""

import json
import os
import time
from pathlib import Path

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset


def load_dataset_from_jsonl(dataset_path: str) -> Dataset:
    """Load a JSONL dataset and format it for SFTTrainer."""
    samples = []
    with open(dataset_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            # Convert ChatML messages to a single text string
            text = ""
            for msg in entry["messages"]:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    text += f"<|system|>\n{content}</s>\n"
                elif role == "user":
                    text += f"<|user|>\n{content}</s>\n"
                elif role == "assistant":
                    text += f"<|assistant|>\n{content}</s>\n"
            samples.append({"text": text})
    return Dataset.from_list(samples)


class MetricsCallback:
    """Writes metrics to a JSON file during training."""

    def __init__(self, metrics_path: str, status_path: str):
        self.metrics_path = metrics_path
        self.status_path = status_path
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        elapsed = time.time() - self.start_time
        progress = state.global_step / state.max_steps if state.max_steps > 0 else 0
        metrics = {
            "train_loss": logs.get("loss", 0),
            "eval_loss": logs.get("eval_loss", 0),
            "duration": f"{elapsed:.1f}s",
            "progress": round(progress, 4),
            "step": state.global_step,
            "max_steps": state.max_steps,
        }
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f)

    def on_train_end(self, args, state, control, **kwargs):
        self._write_status("done")

    def _write_status(self, status: str):
        with open(self.status_path, "w") as f:
            json.dump({"status": status}, f)


def run_training(
    dataset_path: str,
    base_model: str,
    output_dir: str,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
) -> dict:
    """Run LoRA fine-tuning using Unsloth.

    Args:
        dataset_path: Path to JSONL training data.
        base_model: Path to base model (HF format or local path).
        output_dir: Directory to save the LoRA adapter.
        epochs: Number of training epochs.
        learning_rate: Learning rate for training.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha.
        batch_size: Training batch size.

    Returns:
        Dictionary with training metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.json")
    status_path = os.path.join(output_dir, "status.json")

    # Write initial status
    with open(status_path, "w") as f:
        json.dump({"status": "training"}, f)

    start_time = time.time()

    # Load model with Unsloth (4-bit quantization)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    dataset = load_dataset_from_jsonl(dataset_path)

    # Set up training
    adapter_dir = os.path.join(output_dir, "adapter")
    training_args = TrainingArguments(
        output_dir=adapter_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_8bit",
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    callback = MetricsCallback(metrics_path, status_path)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
        callbacks=[callback],
    )

    # Train
    train_result = trainer.train()

    # Save the adapter
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    elapsed = time.time() - start_time
    metrics = {
        "train_loss": train_result.training_loss,
        "duration": f"{elapsed:.1f}s",
        "samples_used": len(dataset),
        "progress": 1.0,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    with open(status_path, "w") as f:
        json.dump({"status": "done"}, f)

    return metrics
