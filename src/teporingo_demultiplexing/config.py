"""Configuration loading helpers for Teporingo-Demultiplexing."""

from __future__ import annotations

from pathlib import Path


def _coerce_scalar(value: str):
    value = value.strip()

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()

    lower_value = value.lower()

    if lower_value in {"true", "false"}:
        return lower_value == "true"

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path: str | Path) -> dict:
    """Load a minimal YAML subset used by the scaffold config.

    Supports nested mappings created by two-space indentation.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(0, root)]

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if ":" not in stripped:
            raise ValueError(f"Invalid key/value line in {config_path}: {raw_line!r}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent < stack[-1][0]:
            stack.pop()

        if not stack:
            raise ValueError(f"Invalid indentation in {config_path}: {raw_line!r}")

        current_section = stack[-1][1]

        if not value:
            nested_section: dict[str, object] = {}
            current_section[key] = nested_section
            stack.append((indent + 2, nested_section))
            continue

        current_section[key] = _coerce_scalar(value)

    return root
