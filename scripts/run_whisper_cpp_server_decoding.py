"""whisper.cpp server로 변환된 GGML 모델을 eval manifest에 대해 디코딩한다."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


LOGGER = logging.getLogger("run_whisper_cpp_server_decoding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuned GGML 모델을 whisper.cpp server로 디코딩한다.")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path, default=Path("data/whisper_small_lora/eval.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--result_dir", dest="output_dir", type=Path, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--server_binary_path", type=Path, default=Path("third_party/whisper.cpp/build/bin/whisper-server"))
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--model", dest="model_name", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--processors", type=int, default=1)
    parser.add_argument("--request_timeout_seconds", type=float, default=None)
    parser.add_argument("--server_start_timeout_seconds", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--retry_errors", action="store_true")
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--no_evaluate", action="store_true")
    parser.add_argument(
        "--treat_replacement_as_error",
        action="store_true",
        help="Treat U+FFFD replacement characters in server text as controlled decode errors.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        parser.error("--output_dir is required")
    return args


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def infer_quantization(model_path: Path) -> str:
    name = model_path.name
    for quantization in ("q5_0", "q8_0", "q5_1", "q4_0", "q4_1"):
        if quantization in name:
            return quantization
    return "f16"


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    model_path = resolve_path(args.model_path)
    output_dir = resolve_path(args.output_dir)
    quantization = args.quantization or infer_quantization(model_path)
    model_name = args.model_name or model_path.parent.name
    experiment = f"{model_name}_{quantization}_beam{args.beam_size}_whisper_cpp_server"
    return {
        "engine": "whisper_cpp_server",
        "experiment": experiment,
        "model": model_name,
        "model_path": str(model_path),
        "beam_size": args.beam_size,
        "quantization": quantization,
        "precision": quantization,
        "manifest_path": str(resolve_path(args.manifest_path)),
        "result_root": str(output_dir.parent),
        "result_dir": str(output_dir),
        "output_root": str(output_dir.parent),
        "output_dir": str(output_dir),
        "project_root": str(PROJECT_ROOT),
        "server_binary_path": str(resolve_path(args.server_binary_path)),
        "host": args.host,
        "port": args.port,
        "device": args.device,
        "device_index": args.device_index,
        "language": args.language,
        "decode_defaults": {
            "threads": args.threads,
            "processors": args.processors,
            "temperature": 0.0,
            "temperature_inc": 0.0,
            "no_language_probabilities": True,
            "flash_attn": False,
        },
        "request_timeout_seconds": args.request_timeout_seconds,
        "server_start_timeout_seconds": args.server_start_timeout_seconds,
        "warmup": not args.no_warmup,
        "treat_replacement_as_error": args.treat_replacement_as_error,
    }


def evaluate_if_needed(args: argparse.Namespace, manifest_path: Path, result_dir: Path) -> None:
    if args.no_evaluate:
        LOGGER.info("Evaluation skipped by --no_evaluate")
        return
    if args.num_shards != 1:
        LOGGER.info("Shard mode detected. Evaluate after all shards finish:")
        LOGGER.info("python scripts/evaluate_predictions.py --manifest_path %s --output_dir %s", manifest_path, result_dir)
        return

    from core.metrics import evaluate_result_dir

    evaluate_result_dir(manifest_path, result_dir)
    LOGGER.info("Wrote metrics: %s", result_dir / "metrics.json")


def main() -> None:
    from core.logging_utils import setup_logging
    from runners.whisper_cpp_server import run_whisper_cpp_server

    args = parse_args()
    config = build_config(args)
    result_dir = Path(config["result_dir"])
    manifest_path = Path(config["manifest_path"])
    setup_logging(result_dir / "logs" / "run.log")

    LOGGER.info("=== whisper.cpp server decoding started ===")
    LOGGER.info("model_path=%s manifest=%s", config["model_path"], config["manifest_path"])
    run_whisper_cpp_server(config, args)
    evaluate_if_needed(args, manifest_path, result_dir)
    LOGGER.info("Output dir: %s", result_dir)
    LOGGER.info("=== DONE whisper.cpp server decoding ===")


if __name__ == "__main__":
    main()
