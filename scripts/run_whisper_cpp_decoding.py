"""whisper.cpp ggml 모델을 fine-tuning eval manifest로 디코딩하고 평가한다."""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DOMAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOMAIN_ROOT / "src"))


LOGGER = logging.getLogger("run_whisper_cpp_decoding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="whisper.cpp ggml 모델 fine-tuning eval decoding 스크립트")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=Path("data/whisper_small_lora/eval.jsonl"),
    )
    parser.add_argument("--output_root", type=Path, default=Path("exp/results"))
    parser.add_argument("--result_root", dest="output_root", type=Path, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--result_dir", dest="output_dir", type=Path, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--whisper_cpp_dir", type=Path, default=DOMAIN_ROOT / "third_party" / "whisper.cpp")
    parser.add_argument("--language", default="ko")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--gpu_device", dest="device_index", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--no_gpu", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--retry_errors", action="store_true")
    parser.add_argument("--no_evaluate", action="store_true")
    return parser.parse_args()


def resolve_required_path(path: Path) -> Path:
    from finetuning.config import resolve_project_path

    resolved = resolve_project_path(DOMAIN_ROOT, path)
    if resolved is None:
        raise ValueError("Path must not be None")
    return resolved


def default_output_dir(output_root: Path, model_path: Path, beam_size: int) -> Path:
    converted_name = model_path.parent.name
    model_name = model_path.stem.replace("ggml-model", "ggml").strip("-")
    return output_root / f"{converted_name}_whisper_cpp_{model_name}_beam{beam_size}"


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    from whisper_cpp.runtime import find_cli_binary

    model_path = resolve_required_path(args.model_path)
    manifest_path = resolve_required_path(args.manifest_path)
    output_root = resolve_required_path(args.output_root)
    output_dir = (
        resolve_required_path(args.output_dir)
        if args.output_dir is not None
        else default_output_dir(output_root, model_path, args.beam_size)
    )
    device = "cpu" if args.no_gpu else args.device
    whisper_cpp_dir = resolve_required_path(args.whisper_cpp_dir)
    whisper_cli = find_cli_binary(whisper_cpp_dir)
    if not model_path.is_file():
        raise FileNotFoundError(f"whisper.cpp model not found: {model_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    return {
        "engine": "whisper_cpp",
        "runner": "whisper_cpp_cli",
        "experiment": output_dir.name,
        "model_path": str(model_path),
        "whisper_cli": str(whisper_cli),
        "whisper_cpp_dir": str(whisper_cpp_dir),
        "beam_size": args.beam_size,
        "manifest_path": str(manifest_path),
        "result_root": str(output_root),
        "result_dir": str(output_dir),
        "output_root": str(output_root),
        "output_dir": str(output_dir),
        "language": args.language,
        "device": device,
        "device_index": args.device_index,
        "use_gpu": device == "cuda",
        "gpu_device": args.device_index,
        "threads": args.threads,
        "decode_options": {
            "output_json": True,
            "no_prints": True,
        },
    }


def temp_output_base(result_dir: Path, sample_id: str) -> Path:
    safe_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in sample_id)
    return result_dir / "whisper_cpp_json" / safe_id


def parse_time_offset_ms(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return None


def parse_whisper_cpp_json(path: Path) -> tuple[str, list[dict[str, Any]]]:
    from decoding.errors import ControlledDecodeError

    try:
        raw_json = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ControlledDecodeError(
            "whisper_cpp_json_invalid_utf8",
            "whisper.cpp JSON output is not valid UTF-8 "
            f"(json_path={path}, byte_position={exc.start}, details={exc.reason})",
        ) from None

    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ControlledDecodeError(
            "whisper_cpp_json_invalid",
            "whisper.cpp JSON output is malformed "
            f"(json_path={path}, line={exc.lineno}, column={exc.colno}, details={exc.msg})",
        ) from None

    transcription = payload.get("transcription") or []
    if not isinstance(transcription, list):
        raise ControlledDecodeError(
            "whisper_cpp_json_invalid_schema",
            f"whisper.cpp JSON transcription must be a list (json_path={path})",
        )

    segments = []
    texts = []
    for index, segment in enumerate(transcription):
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "")).strip()
        texts.append(text)
        offsets = segment.get("offsets") if isinstance(segment.get("offsets"), dict) else {}
        segments.append(
            {
                "id": index,
                "start": parse_time_offset_ms(offsets.get("from")),
                "end": parse_time_offset_ms(offsets.get("to")),
                "text": text,
            }
        )
    return " ".join(text for text in texts if text).strip(), segments


def build_whisper_cpp_command(config: dict[str, Any], audio_path: Path, output_base: Path) -> list[str]:
    command = [
        config["whisper_cli"],
        "-m",
        config["model_path"],
        "-f",
        str(audio_path),
        "-l",
        config["language"],
        "-bs",
        str(config["beam_size"]),
        "-oj",
        "-of",
        str(output_base),
        "-np",
    ]
    if config.get("threads") is not None:
        command.extend(["-t", str(config["threads"])])
    if config.get("device") == "cuda" and config.get("device_index") is not None:
        command.extend(["-dev", str(config["device_index"])])
    if config.get("device") == "cpu":
        command.append("-ng")
    return command


def run_whisper_cli(config: dict[str, Any], item: dict[str, Any], command_env: dict[str, str]) -> tuple[str, list[dict[str, Any]], float]:
    from decoding.audio import prepared_audio_path

    result_dir = Path(config["result_dir"])
    output_base = temp_output_base(result_dir, str(item["id"]))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_base.with_suffix(".json")
    if json_path.exists():
        json_path.unlink()

    with prepared_audio_path(item, project_root=DOMAIN_ROOT) as audio_path:
        command = build_whisper_cpp_command(config, audio_path, output_base)
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=DOMAIN_ROOT,
            env=command_env,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    decode_time = time.perf_counter() - start
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-40:])
        raise RuntimeError(f"whisper.cpp decode failed with exit code {completed.returncode}\n{tail}")
    if not json_path.exists():
        raise RuntimeError(f"whisper.cpp did not create JSON output: {json_path}")

    prediction_raw, segments = parse_whisper_cpp_json(json_path)
    return prediction_raw, segments, decode_time


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
    from core.io import write_json
    from core.logging_utils import setup_logging
    from decoding.decode_loop import DecodeOutput, decode_rows
    from decoding.run_utils import fail_if_all_samples_failed, finish_run, prepare_decode_run
    from whisper_cpp.runtime import env_with_library_dirs, library_dirs_from_project

    args = parse_args()
    config = build_config(args)
    result_dir = Path(config["result_dir"])
    setup_logging(result_dir / "logs" / "run.log")

    LOGGER.info("=== whisper.cpp decoding started ===")
    LOGGER.info("model=%s manifest=%s", config["model_path"], config["manifest_path"])

    decode_run = prepare_decode_run(config, args)
    write_json(decode_run.run_config_path, decode_run.run_config)
    command_env = env_with_library_dirs(library_dirs_from_project(Path(config["whisper_cpp_dir"])))

    def decode_one(item: dict[str, Any]) -> DecodeOutput:
        prediction_raw, segments, decode_time = run_whisper_cli(config, item, command_env)
        return DecodeOutput(prediction_raw=prediction_raw, segments=segments, decode_time=decode_time)

    decoded_count, error_count = decode_rows(
        rows=decode_run.rows,
        config=config,
        prediction_path=decode_run.prediction_path,
        error_path=decode_run.error_path,
        done_ids=decode_run.done_ids,
        limit=args.limit,
        decode_one=decode_one,
        logger=LOGGER,
    )

    finish_run(decode_run.run_config, decoded_count, error_count)
    write_json(decode_run.run_config_path, decode_run.run_config)
    fail_if_all_samples_failed(decode_run.run_config)
    evaluate_if_needed(args, Path(config["manifest_path"]), result_dir)
    LOGGER.info("Output dir: %s", result_dir)
    LOGGER.info("=== DONE whisper.cpp decoding ===")


if __name__ == "__main__":
    main()
