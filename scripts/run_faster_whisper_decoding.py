"""Decode an eval manifest with a faster-whisper/CTranslate2 model."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


LOGGER = logging.getLogger("run_faster_whisper_decoding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="faster-whisper CTranslate2 model decoding script")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path, default=Path("data/whisper_small_lora/eval.jsonl"))
    parser.add_argument("--output_root", type=Path, default=Path("exp/results"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument(
        "--compute_type",
        choices=("float16", "float32", "int8", "int8_float16", "int8_float32", "int16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--vad_filter", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--retry_errors", action="store_true")
    parser.add_argument("--no_evaluate", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_output_dir(output_root: Path, model_dir: Path, compute_type: str, beam_size: int) -> Path:
    return output_root / f"{model_dir.name}_faster_whisper_{compute_type}_beam{beam_size}"


def build_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    model_dir = resolve_path(args.model_dir)
    manifest_path = resolve_path(args.manifest_path)
    output_root = resolve_path(args.output_root)
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(output_root, model_dir, args.compute_type, args.beam_size)

    if not model_dir.is_dir():
        raise FileNotFoundError(f"faster-whisper model directory not found: {model_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    config = {
        "engine": "faster_whisper",
        "runner": "faster_whisper",
        "experiment": output_dir.name,
        "model": str(model_dir),
        "model_dir": str(model_dir),
        "beam_size": args.beam_size,
        "precision": args.compute_type,
        "compute_type": args.compute_type,
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "output_dir": str(output_dir),
        "result_root": str(output_root),
        "result_dir": str(output_dir),
        "device": args.device,
        "device_index": args.device_index,
        "language": args.language,
        "vad_filter": args.vad_filter,
    }
    return config, manifest_path


def decode_segments(model: Any, item: dict[str, Any], config: dict[str, Any]) -> tuple[str, list[dict[str, Any]], float]:
    from decoding.audio import prepared_audio_path

    with prepared_audio_path(item, project_root=PROJECT_ROOT) as audio_path:
        start = time.perf_counter()
        segments_iter, _info = model.transcribe(
            str(audio_path),
            language=config["language"],
            task="transcribe",
            beam_size=int(config["beam_size"]),
            vad_filter=bool(config.get("vad_filter", False)),
        )
        segments = []
        texts = []
        for index, segment in enumerate(segments_iter):
            text = str(segment.text or "").strip()
            texts.append(text)
            segments.append(
                {
                    "id": index,
                    "start": segment.start,
                    "end": segment.end,
                    "text": text,
                }
            )
    decode_time = time.perf_counter() - start
    return " ".join(text for text in texts if text).strip(), segments, decode_time


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
    from core.io import write_json
    from decoding.decode_loop import DecodeOutput, decode_rows
    from decoding.run_utils import fail_if_all_samples_failed, finish_run, prepare_decode_run
    from faster_whisper import WhisperModel

    args = parse_args()
    config, manifest_path = build_config(args)
    result_dir = Path(config["result_dir"])
    setup_logging(result_dir / "logs" / "run.log")

    LOGGER.info("=== faster-whisper decoding started ===")
    LOGGER.info("model=%s manifest=%s", config["model_dir"], config["manifest_path"])

    decode_run = prepare_decode_run(config, args)
    write_json(decode_run.run_config_path, decode_run.run_config)

    model = WhisperModel(
        config["model_dir"],
        device=config["device"],
        device_index=int(config["device_index"]),
        compute_type=config["compute_type"],
    )

    def decode_one(item: dict[str, Any]) -> DecodeOutput:
        prediction_raw, segments, decode_time = decode_segments(model, item, config)
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
    evaluate_if_needed(args, manifest_path, result_dir)
    LOGGER.info("Output dir: %s", result_dir)
    LOGGER.info("=== DONE faster-whisper decoding ===")


if __name__ == "__main__":
    main()
