"""实验产物的文件系统和元数据辅助函数。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def ensure_new_output_dir(
    path: Path, resume_from_checkpoint: Path | None = None
) -> None:
    """创建新输出目录，或校验显式指定的 resume 目标。"""
    if path.exists():
        if resume_from_checkpoint is None:
            raise FileExistsError(f"output directory already exists: {path}")
        try:
            resume_from_checkpoint.resolve().relative_to(path.resolve())
        except ValueError as exc:
            raise ValueError(
                "resume checkpoint must be inside the output directory"
            ) from exc
        if not resume_from_checkpoint.is_dir():
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {resume_from_checkpoint}"
            )
        return
    if resume_from_checkpoint is not None:
        raise FileNotFoundError(
            f"cannot resume because output directory does not exist: {path}"
        )
    path.mkdir(parents=True)


def package_versions(packages: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None
