from typing import Any


REPLACEMENT_CHAR = "\ufffd"


def replacement_char_count(text: object) -> int:
    return str(text or "").count(REPLACEMENT_CHAR)


def segment_replacement_char_count(segments: list[dict[str, Any]]) -> int:
    return sum(replacement_char_count(segment.get("text", "")) for segment in segments)


def prediction_replacement_char_count(prediction_raw: object, segments: list[dict[str, Any]] | None = None) -> int:
    count = replacement_char_count(prediction_raw)
    if count == 0 and segments:
        count = segment_replacement_char_count(segments)
    return count

