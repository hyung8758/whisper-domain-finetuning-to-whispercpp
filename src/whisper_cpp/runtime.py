import os
from pathlib import Path


class WhisperCppBinaryNotFoundError(FileNotFoundError):
    pass


def library_dirs_from_build_dir(build_dir: Path) -> list[Path]:
    candidates = [
        build_dir / "src",
        build_dir / "ggml" / "src",
        build_dir / "ggml" / "src" / "ggml-cuda",
    ]
    return [path for path in candidates if path.exists()]


def library_dirs_from_project(whisper_cpp_dir: Path) -> list[Path]:
    return library_dirs_from_build_dir(whisper_cpp_dir / "build")


def find_first_existing(candidates: tuple[Path, ...], description: str, hint: str) -> Path:
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise WhisperCppBinaryNotFoundError(
        f"{description} not found.\n"
        f"{hint}\n\n"
        "Expected one of:\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


def find_quantize_binary(whisper_cpp_dir: Path) -> Path:
    return find_first_existing(
        (
            whisper_cpp_dir / "build" / "bin" / "whisper-quantize",
            whisper_cpp_dir / "build" / "bin" / "quantize",
        ),
        description="whisper.cpp quantize binary",
        hint="Build whisper.cpp first, then run conversion again.",
    )


def find_cli_binary(whisper_cpp_dir: Path) -> Path:
    return find_first_existing(
        (
            whisper_cpp_dir / "build" / "bin" / "whisper-cli",
            whisper_cpp_dir / "build" / "bin" / "main",
        ),
        description="whisper.cpp CLI binary",
        hint="Build whisper.cpp first, then run decoding again.",
    )


def env_with_library_dirs(library_dirs: list[Path]) -> dict[str, str]:
    env = dict(os.environ)
    values = [str(path) for path in library_dirs]
    current = env.get("LD_LIBRARY_PATH")
    if current:
        values.append(current)
    env["LD_LIBRARY_PATH"] = ":".join(values)
    return env
