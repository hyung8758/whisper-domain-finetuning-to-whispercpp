"""Diagnose UTF-8 validity of whisper.cpp CLI JSON output without printing text payloads."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="whisper.cpp JSON UTF-8 diagnostic runner")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--whisper_cpp_dir", type=Path, default=PROJECT_ROOT / "third_party" / "whisper.cpp")
    parser.add_argument("--whisper_cli_path", type=Path, default=None)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def safe_output_base(output_dir: Path, sample_id: object) -> Path:
    safe_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(sample_id)).strip("_")
    return output_dir / "whisper_cpp_json" / (safe_id or "sample")


def build_command(args: argparse.Namespace, cli_path: Path, audio_path: Path, output_base: Path) -> list[str]:
    command = [
        str(cli_path),
        "-m",
        str(resolve_path(args.model_path)),
        "-f",
        str(audio_path),
        "-l",
        args.language,
        "-bs",
        str(args.beam_size),
        "-oj",
        "-of",
        str(output_base),
        "-np",
    ]
    if args.threads is not None:
        command.extend(["-t", str(args.threads)])
    if args.device == "cpu":
        command.append("-ng")
    else:
        command.extend(["-dev", str(args.device_index)])
    return command


def read_json_bytes(json_path: Path) -> dict[str, Any]:
    from decoding.utf8 import replacement_char_count

    data = json_path.read_bytes()
    result: dict[str, Any] = {
        "json_path": str(json_path),
        "json_file_size": len(data),
        "json_utf8_valid": True,
        "json_parse_ok": False,
        "utf8_error_position": None,
        "utf8_error_reason": None,
        "replacement_char_count": 0,
    }
    try:
        raw_text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        result.update(
            {
                "json_utf8_valid": False,
                "utf8_error_position": exc.start,
                "utf8_error_reason": exc.reason,
            }
        )
        raw_text = data.decode("utf-8", errors="replace")

    result["replacement_char_count"] = replacement_char_count(raw_text)
    try:
        json.loads(raw_text)
        result["json_parse_ok"] = True
    except json.JSONDecodeError as exc:
        result["json_error_line"] = exc.lineno
        result["json_error_column"] = exc.colno
        result["json_error_reason"] = exc.msg
    return result


def append_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def main() -> None:
    from core.io import read_jsonl, write_json
    from decoding.audio import prepared_audio_path
    from whisper_cpp.runtime import (
        env_with_library_dirs,
        find_cli_binary,
        library_dirs_from_build_dir,
        library_dirs_from_project,
    )

    args = parse_args()
    model_path = resolve_path(args.model_path)
    manifest_path = resolve_path(args.manifest_path)
    output_dir = resolve_path(args.output_dir)
    whisper_cpp_dir = resolve_path(args.whisper_cpp_dir)
    cli_path = resolve_path(args.whisper_cli_path) if args.whisper_cli_path else find_cli_binary(whisper_cpp_dir)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if output_dir.exists() and args.overwrite:
        for path in (output_dir / "diagnostics.jsonl", output_dir / "summary.json"):
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "whisper_cpp_json").mkdir(parents=True, exist_ok=True)

    if cli_path.parent.name == "bin" and cli_path.parent.parent.name == "build":
        library_dirs = library_dirs_from_build_dir(cli_path.parent.parent)
    else:
        library_dirs = library_dirs_from_project(whisper_cpp_dir)
    command_env = env_with_library_dirs(library_dirs)

    rows = read_jsonl(manifest_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    summary = {
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "whisper_cli_path": str(cli_path),
        "samples": 0,
        "returncode_failures": 0,
        "missing_json": 0,
        "invalid_utf8_json": 0,
        "json_parse_failures": 0,
        "replacement_char_rows": 0,
        "replacement_char_count": 0,
    }

    with (output_dir / "diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for item in rows:
            output_base = safe_output_base(output_dir, item["id"])
            json_path = output_base.with_suffix(".json")
            if json_path.exists():
                json_path.unlink()

            start = time.perf_counter()
            row = {
                "id": item["id"],
                "dataset": item.get("dataset"),
                "bucket": item.get("bucket"),
                "duration": item.get("duration"),
            }
            try:
                with prepared_audio_path(item, project_root=PROJECT_ROOT) as audio_path:
                    command = build_command(args, cli_path, audio_path, output_base)
                    completed = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        env=command_env,
                        check=False,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                row["decode_time"] = round(time.perf_counter() - start, 6)
                row["returncode"] = completed.returncode
                if completed.returncode != 0:
                    row["failure"] = "returncode"
                    row["stdout_tail"] = "\n".join((completed.stdout or "").splitlines()[-20:])
                    summary["returncode_failures"] += 1
                elif not json_path.exists():
                    row["failure"] = "missing_json"
                    summary["missing_json"] += 1
                else:
                    row.update(read_json_bytes(json_path))
                    if not row["json_utf8_valid"]:
                        summary["invalid_utf8_json"] += 1
                    if not row["json_parse_ok"]:
                        summary["json_parse_failures"] += 1
                    if row["replacement_char_count"] > 0:
                        summary["replacement_char_rows"] += 1
                        summary["replacement_char_count"] += row["replacement_char_count"]
            except Exception as exc:
                row["decode_time"] = round(time.perf_counter() - start, 6)
                row["failure"] = "exception"
                row["error"] = str(exc)
                summary["returncode_failures"] += 1

            summary["samples"] += 1
            append_jsonl(handle, row)

    write_json(output_dir / "summary.json", summary)
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
