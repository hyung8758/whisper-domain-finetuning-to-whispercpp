import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from core.io import append_jsonl, write_json
from decoding.audio import prepared_audio_path
from decoding.run_utils import (
    fail_if_all_samples_failed,
    finish_run,
    make_error_row,
    make_prediction_row,
    prepare_decode_run,
)
from whisper_cpp.runtime import env_with_library_dirs, library_dirs_from_build_dir


LOGGER = logging.getLogger(__name__)


def validate_runtime(config: dict[str, Any]) -> None:
    server_binary_path = Path(config["server_binary_path"])
    model_path = Path(config["model_path"])
    if not server_binary_path.is_file():
        raise FileNotFoundError(f"whisper.cpp server binary not found: {server_binary_path}")
    if not os.access(server_binary_path, os.X_OK):
        raise PermissionError(f"whisper.cpp server binary is not executable: {server_binary_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"whisper.cpp model not found: {model_path}")


def server_command(config: dict[str, Any]) -> list[str]:
    defaults = config.get("decode_defaults", {})
    command = [
        config["server_binary_path"],
        "--model",
        config["model_path"],
        "--host",
        config["host"],
        "--port",
        str(config["port"]),
        "--language",
        config["language"],
        "--beam-size",
        str(config["beam_size"]),
        "--threads",
        str(defaults.get("threads", 4)),
        "--processors",
        str(defaults.get("processors", 1)),
    ]
    if config["device"] == "cpu":
        command.append("--no-gpu")
    else:
        command.extend(["--device", str(config["device_index"])])

    flash_attn = defaults.get("flash_attn")
    if flash_attn is True:
        command.append("--flash-attn")
    elif flash_attn is False:
        command.append("--no-flash-attn")
    if bool(defaults.get("no_fallback", False)):
        command.append("--no-fallback")
    if bool(defaults.get("no_language_probabilities", True)):
        command.append("--no-language-probabilities")
    command.extend(str(arg) for arg in defaults.get("server_extra_args", []))
    return command


def request_fields(config: dict[str, Any]) -> dict[str, str]:
    defaults = config.get("decode_defaults", {})
    fields = {
        "response_format": "verbose_json",
        "temperature": str(defaults.get("temperature", 0.0)),
        "temperature_inc": str(defaults.get("temperature_inc", 0.0)),
        "beam_size": str(config["beam_size"]),
        "language": config["language"],
        "no_language_probabilities": "true",
    }
    if defaults.get("no_fallback", False):
        fields["no_fallback"] = "true"
    for key, value in defaults.get("request_fields", {}).items():
        fields[str(key)] = str(value)
    return fields


def wait_for_server(config: dict[str, Any], process: subprocess.Popen) -> None:
    deadline = time.monotonic() + float(config.get("server_start_timeout_seconds") or 120)
    url = f"http://{config['host']}:{config['port']}/health"
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"whisper-server exited early with code {process.returncode}")
        try:
            response = requests.get(url, timeout=2.0)
            if response.status_code == 200 and response.json().get("status") == "ok":
                return
            last_error = response.text
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"whisper-server did not become ready at {url}. Last error: {last_error}")


def whisper_cpp_env(config: dict[str, Any]) -> dict[str, str]:
    build_dir = Path(config["server_binary_path"]).resolve().parents[1]
    return env_with_library_dirs(library_dirs_from_build_dir(build_dir))


def start_server(config: dict[str, Any], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = server_command(config)
    LOGGER.info("Starting whisper-server: %s", " ".join(command))
    log_file = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=whisper_cpp_env(config))
    except Exception:
        log_file.close()
        raise

    process._stt_log_file = log_file
    try:
        wait_for_server(config, process)
    except Exception:
        stop_server(process)
        raise
    return process


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    log_file = getattr(process, "_stt_log_file", None)
    if log_file is not None:
        log_file.close()


def parse_response(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]], float | None]:
    prediction_raw = str(data.get("text", "")).strip()
    segments = []
    for index, segment in enumerate(data.get("segments", [])):
        segments.append(
            {
                "id": segment.get("id", index),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": str(segment.get("text", "")).strip(),
            }
        )
    if not prediction_raw:
        prediction_raw = " ".join(segment["text"] for segment in segments).strip()
    timings = data.get("timings", {})
    inference_sec = timings.get("inference_sec")
    return prediction_raw, segments, float(inference_sec) if inference_sec is not None else None


def transcribe(
    server_url: str,
    item: dict[str, Any],
    fields: dict[str, str],
    timeout: float | None,
    project_root: Path | None = None,
) -> tuple[dict[str, Any], float]:
    with prepared_audio_path(item, project_root=project_root) as audio_path:
        start = time.perf_counter()
        with audio_path.open("rb") as audio_file:
            files = {"file": (audio_path.name, audio_file, "audio/wav")}
            response = requests.post(server_url, data=fields, files=files, timeout=timeout)
    request_time = time.perf_counter() - start
    response.raise_for_status()
    return response.json(), request_time


def warmup_server(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if not config.get("warmup") or not rows:
        return
    server_url = f"http://{config['host']}:{config['port']}/inference"
    project_root = Path(config["project_root"]) if config.get("project_root") else None
    try:
        transcribe(server_url, rows[0], request_fields(config), config.get("request_timeout_seconds"), project_root)
        LOGGER.info("Warmup finished with id=%s", rows[0]["id"])
    except Exception:
        LOGGER.exception("Warmup failed; eval decoding will continue")


def decode_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    prediction_path: Path,
    error_path: Path,
    done_ids: set[str],
    limit: int | None,
) -> tuple[int, int]:
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    server_url = f"http://{config['host']}:{config['port']}/inference"
    fields = request_fields(config)
    project_root = Path(config["project_root"]) if config.get("project_root") else None
    decoded_count = 0
    error_count = 0

    with prediction_path.open("a", encoding="utf-8") as prediction_file, error_path.open(
        "a", encoding="utf-8"
    ) as error_file:
        for item in tqdm(rows, desc="Decoding"):
            if item["id"] in done_ids:
                continue
            if limit is not None and decoded_count + error_count >= limit:
                break

            start = time.perf_counter()
            try:
                data, request_time = transcribe(
                    server_url,
                    item,
                    fields,
                    config.get("request_timeout_seconds"),
                    project_root,
                )
                prediction_raw, segments, inference_sec = parse_response(data)
                backend_time = inference_sec if inference_sec is not None else request_time
                row = make_prediction_row(item, prediction_raw, segments, backend_time, config)
                row["request_time"] = round(request_time, 6)
                row["backend_inference_time"] = round(backend_time, 6)
                row["timing_source"] = "server_timings" if inference_sec is not None else "http_request"
                append_jsonl(prediction_file, row)
                decoded_count += 1
            except Exception as exc:
                request_time = time.perf_counter() - start
                append_jsonl(error_file, make_error_row(item, "decode_failed", str(exc), request_time, config))
                LOGGER.exception("Decode failed for id=%s", item["id"])
                error_count += 1

    return decoded_count, error_count


def run_whisper_cpp_server(config: dict[str, Any], args) -> None:
    validate_runtime(config)
    decode_run = prepare_decode_run(config, args)
    write_json(decode_run.run_config_path, decode_run.run_config)

    LOGGER.info(
        "Running whisper.cpp server model=%s device=%s device_index=%s port=%s",
        config["model_path"],
        config["device"],
        config["device_index"],
        config["port"],
    )
    LOGGER.info("Experiment=%s beam_size=%s quantization=%s", config["experiment"], config["beam_size"], config["quantization"])
    LOGGER.info("Shard %s/%s has %s samples", args.shard_index, args.num_shards, len(decode_run.rows))

    server = None
    try:
        server_log_path = decode_run.log_path.parent / f"server.shard_{args.shard_index:03d}.log"
        server = start_server(config, server_log_path)
        warmup_server(config, decode_run.rows)
        decoded_count, error_count = decode_rows(
            rows=decode_run.rows,
            config=config,
            prediction_path=decode_run.prediction_path,
            error_path=decode_run.error_path,
            done_ids=decode_run.done_ids,
            limit=args.limit,
        )
    finally:
        stop_server(server)

    finish_run(decode_run.run_config, decoded_count, error_count)
    write_json(decode_run.run_config_path, decode_run.run_config)
    fail_if_all_samples_failed(decode_run.run_config)
    LOGGER.info("Wrote predictions to %s", decode_run.prediction_path)
    LOGGER.info("Wrote errors to %s", decode_run.error_path)
