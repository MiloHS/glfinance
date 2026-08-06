"""Fail when non-public artifacts or likely secrets are tracked by Git."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys


FORBIDDEN_PREFIXES = (
    ".venv/",
    "Finance_Graph_Learning/",
    "data/",
    "experiments/",
    "legacy/",
    "results/",
)
FORBIDDEN_FILES = {
    "GLFinance_notebook.ipynb",
    "GFinance_Portfolio_notebook.ipynb",
    "review.txt",
}
FORBIDDEN_SUFFIXES = {
    ".csv",
    ".key",
    ".lic",
    ".npy",
    ".npz",
    ".parquet",
    ".pem",
    ".pickle",
    ".pkl",
}
ALLOWED_NOTEBOOK = "notebooks/portfolio_walk_forward.ipynb"
MAX_FILE_BYTES = 2_000_000
SECRET_PATTERNS = {
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "OpenAI-style key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "private key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "assigned credential": re.compile(
        rb"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
        rb"\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']"
    ),
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        Path(item.decode("utf-8"))
        for item in result.stdout.split(b"\0")
        if item
    ]


def check_notebook(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"{path}: cannot parse notebook: {error}"]
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            errors.append(f"{path}: code cell {index} has an execution count")
        if cell.get("outputs"):
            errors.append(f"{path}: code cell {index} contains saved output")
    return errors


def main() -> int:
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    )
    errors: list[str] = []
    for relative in tracked_files():
        name = relative.as_posix()
        path = root / relative
        if name in FORBIDDEN_FILES or name.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"{name}: private or generated path must not be tracked")
        if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"{name}: data, credential, or binary artifact is forbidden")
        if relative.suffix.lower() == ".ipynb" and name != ALLOWED_NOTEBOOK:
            errors.append(f"{name}: only {ALLOWED_NOTEBOOK} may be tracked")
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"{name}: exceeds {MAX_FILE_BYTES:,} bytes")
        if not path.is_file():
            continue
        content = path.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"{name}: possible {label}")
        if name == ALLOWED_NOTEBOOK:
            errors.extend(check_notebook(path))

    if errors:
        print("Repository safety check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Repository safety check passed for {len(tracked_files())} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
