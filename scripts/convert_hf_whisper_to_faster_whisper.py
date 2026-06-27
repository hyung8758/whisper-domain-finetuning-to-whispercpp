"""Convert a local Hugging Face Whisper model directory to faster-whisper/CTranslate2."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HF Whisper model to CTranslate2 for faster-whisper")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--quantization",
        choices=("float16", "float32", "int8", "int8_float16", "int8_float32", "int16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_preprocessor_config(model_dir: Path) -> None:
    preprocessor_path = model_dir / "preprocessor_config.json"
    if preprocessor_path.exists():
        return

    processor_path = model_dir / "processor_config.json"
    if not processor_path.exists():
        raise FileNotFoundError(
            "preprocessor_config.json is missing and processor_config.json was not found. "
            f"model_dir={model_dir}"
        )

    processor = json.loads(processor_path.read_text(encoding="utf-8"))
    feature_extractor = processor.get("feature_extractor")
    if not isinstance(feature_extractor, dict):
        raise ValueError(f"processor_config.json does not contain feature_extractor: {processor_path}")

    preprocessor_path.write_text(
        json.dumps(feature_extractor, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    ensure_preprocessor_config(model_dir)

    converter = shutil.which("ct2-transformers-converter")
    if converter is None:
        raise FileNotFoundError(
            "ct2-transformers-converter not found. Install faster-whisper/ctranslate2 dependencies first."
        )

    command = [
        converter,
        "--model",
        str(model_dir),
        "--output_dir",
        str(output_dir),
        "--copy_files",
        "tokenizer.json",
        "preprocessor_config.json",
        "--quantization",
        args.quantization,
    ]
    if args.force:
        command.append("--force")
    if args.low_cpu_mem_usage:
        command.append("--low_cpu_mem_usage")

    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(output_dir)


if __name__ == "__main__":
    main()
