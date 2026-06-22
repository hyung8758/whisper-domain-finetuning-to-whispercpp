import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from core.io import write_json


DEFAULT_CONFIG: dict[str, Any] = {
    "base_model_name_or_path": "openai/whisper-small",
    "language": "Korean",
    "task": "transcribe",
    "seed": 42,
    "sampling_rate": 16000,
    "device": "cuda",
    "device_index": None,
    "train_path": "data/whisper_small_lora/train.jsonl",
    "dev_path": "data/whisper_small_lora/dev.jsonl",
    "eval_path": "data/whisper_small_lora/eval.jsonl",
    "output_root": "exp/train",
    "exp_name": None,
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
    },
    "training": {
        "epoch": 3,
        "learning_rate": 0.0001,
        "warmup_steps": 200,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": False,
        "fp16": True,
        "bf16": False,
        "logging_steps": 50,
        "save_total_limit": None,
        "dataloader_num_workers": 2,
        "generation_max_length": 225,
        "predict_with_generate": False,
        "resume_from_checkpoint": None,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_finetuning_config(config_path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return deep_update(DEFAULT_CONFIG, loaded)


def resolve_project_path(project_root: Path, path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def make_train_dir(config: dict[str, Any], project_root: Path) -> Path:
    output_root = resolve_project_path(project_root, config["output_root"])
    assert output_root is not None
    exp_name = config.get("exp_name")
    if not exp_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = str(config["base_model_name_or_path"]).split("/")[-1]
        exp_name = f"{timestamp}_{model_name}_lora"
    return output_root / str(exp_name)


def write_resolved_config(train_dir: Path, config: dict[str, Any]) -> None:
    write_json(train_dir / "run_config.json", config)
