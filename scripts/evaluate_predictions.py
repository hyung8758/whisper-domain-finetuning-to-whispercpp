import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.logging_utils import setup_logging
from core.metrics import evaluate_result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="predictions.jsonl을 평가하고 metrics.json을 생성한다.")
    parser.add_argument("--manifest_path", type=Path, default=Path("data/whisper_small_lora/eval.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--result_dir", dest="output_dir", type=Path, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output_dir is None:
        parser.error("--output_dir is required")
    return args


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest_path)
    output_dir = resolve_path(args.output_dir)
    setup_logging(output_dir / "logs" / "evaluate.log")
    evaluate_result_dir(manifest_path, output_dir)


if __name__ == "__main__":
    main()
