"""Whisper 모델을 PEFT LoRA로 fine-tuning한다."""

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
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--device_index", default=None)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def is_distributed_run() -> bool:
    return any(name in os.environ for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))


def normalize_device_index(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"all", "none", "null"}:
            return None
        raw_parts = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        raw_parts = [str(part).strip() for part in value]
    else:
        raw_parts = [str(value).strip()]

    indexes = []
    for part in raw_parts:
        if not part:
            continue
        indexes.append(str(int(part)))
    return ",".join(indexes) if indexes else None


def configure_device(config: dict, requested_device: str | None, requested_device_index) -> None:
    device = requested_device or str(config.get("device") or "cuda")
    device_index = normalize_device_index(
        requested_device_index if requested_device_index is not None else config.get("device_index")
    )
    is_distributed = is_distributed_run()

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config["training"]["fp16"] = False
        config["training"]["bf16"] = False
        LOGGER.info("Using CPU training; disabled fp16/bf16.")
    elif not is_distributed and device_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_index
        LOGGER.info("Using configured CUDA device_index=%s", device_index)
    elif not is_distributed:
        import torch

        visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible_gpu_count > 1:
            raise RuntimeError(
                "Multiple CUDA devices are visible in a non-distributed run. "
                "Use torchrun for multi-GPU training, or set --device_index for single-GPU training. "
                f"visible_gpu_count={visible_gpu_count}"
            )
        LOGGER.info("Using single visible CUDA device. CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    else:
        LOGGER.info("Distributed training detected; using torchrun-assigned CUDA devices.")

    config["device"] = device
    config["device_index"] = device_index


def main() -> None:
    args = parse_args()
    setup_logging()

    from finetuning.config import load_finetuning_config

    config = load_finetuning_config(args.config)
    if args.exp_name is not None:
        config["exp_name"] = args.exp_name
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
