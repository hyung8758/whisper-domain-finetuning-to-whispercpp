"""Whisper small 모델을 PEFT LoRA로 fine-tuning한다."""

import argparse
import logging
import os
import sys
from pathlib import Path


DOMAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOMAIN_ROOT / "src"))


LOGGER = logging.getLogger("train_whisper_lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whisper LoRA 파인튜닝 실행 스크립트")
    parser.add_argument(
        "--config",
        type=Path,
        default=DOMAIN_ROOT / "config" / "whisper_small_lora.yaml",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--device_index", type=int, default=None)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_device(config: dict, requested_device: str | None, requested_device_index: int | None) -> None:
    device = requested_device or str(config.get("device") or "cuda")
    device_index = requested_device_index if requested_device_index is not None else int(config.get("device_index") or 0)
    is_distributed = any(name in os.environ for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config["training"]["fp16"] = False
        config["training"]["bf16"] = False
        LOGGER.info("Using CPU training; disabled fp16/bf16.")
    elif not is_distributed and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
        LOGGER.info("Using single CUDA device_index=%s", device_index)

    config["device"] = device
    config["device_index"] = device_index


def main() -> None:
    args = parse_args()
    setup_logging()

    from finetuning.config import load_finetuning_config

    config = load_finetuning_config(args.config)
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    if args.resume_from_checkpoint is not None:
        config["training"]["resume_from_checkpoint"] = str(args.resume_from_checkpoint)
    configure_device(config, args.device, args.device_index)

    from finetuning.trainer import run_whisper_lora_training

    LOGGER.info("Using config: %s", args.config)
    summary = run_whisper_lora_training(config, project_root=DOMAIN_ROOT)
    LOGGER.info("Wrote training summary: %s", summary["training_summary_path"])


if __name__ == "__main__":
    main()
