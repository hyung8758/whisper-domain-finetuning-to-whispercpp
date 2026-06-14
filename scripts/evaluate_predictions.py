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
    parser.add_argument("--result_dir", type=Path, required=True)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest_path)
    result_dir = resolve_path(args.result_dir)
    setup_logging(result_dir / "logs" / "evaluate.log")
    evaluate_result_dir(manifest_path, result_dir)


if __name__ == "__main__":
    main()
