from pathlib import Path
from typing import Any

from core.io import read_json


DEFAULT_CONFIG = {
    "engine": "openai_whisper",
    "manifest_path": "data/benchmark/manifest.jsonl",
    "output_root": "results",
    "result_root": "results",
    "device": "cuda",
    "device_index": 0,
    "language": "ko",
    "decode_defaults": {
        "task": "transcribe",
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "verbose": False,
    },
    "experiments": [],
}


def load_config(path: Path) -> dict[str, Any]:
    config = copy_config(DEFAULT_CONFIG)
    if path.exists():
        user_config = read_json(path)
        for key, value in user_config.items():
            if key == "decode_defaults":
                config["decode_defaults"].update(value)
            else:
                config[key] = value
    return config


def copy_config(config: dict[str, Any]) -> dict[str, Any]:
    copied = {}
    for key, value in config.items():
        if isinstance(value, dict):
            copied[key] = copy_config(value)
        elif isinstance(value, list):
            copied[key] = [copy_config(item) if isinstance(item, dict) else item for item in value]
        else:
            copied[key] = value
    return copied


def find_experiment(config: dict[str, Any], exp_name_filter: str | None) -> dict[str, Any]:
    experiments = config.get("experiments", [])
    if not experiments:
        raise ValueError("Config must contain at least one experiment.")
    if exp_name_filter is None:
        return dict(experiments[0])
    for experiment in experiments:
        if experiment["exp_name"] == exp_name_filter:
            return dict(experiment)
    exp_names = ", ".join(experiment["exp_name"] for experiment in experiments)
    raise ValueError(f"Unknown experiment: {exp_name_filter}. Available: {exp_names}")


def safe_path_part(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def exp_name(experiment: dict[str, Any]) -> str:
    if experiment.get("exp_name"):
        return experiment["exp_name"]
    return f"{experiment['model']}_beam{experiment.get('beam_size', 5)}_{experiment.get('precision', 'fp16')}"


def result_dir_for(config: dict[str, Any], experiment: dict[str, Any]) -> str:
    return str(
        Path(config.get("output_root", config["result_root"]))
        / safe_path_part(config["engine"])
        / safe_path_part(experiment["model"])
        / safe_path_part(exp_name(experiment))
    )


def output_paths(result_dir: Path, num_shards: int, shard_index: int) -> tuple[Path, Path, Path, Path]:
    if num_shards == 1:
        return (
            result_dir / "predictions.jsonl",
            result_dir / "errors.jsonl",
            result_dir / "run_config.json",
            result_dir / "logs" / "run.log",
        )

    shard_dir = result_dir / "shards"
    suffix = f"shard_{shard_index:03d}"
    return (
        shard_dir / f"predictions.{suffix}.jsonl",
        shard_dir / f"errors.{suffix}.jsonl",
        shard_dir / f"run_config.{suffix}.json",
        result_dir / "logs" / f"run.{suffix}.log",
    )
