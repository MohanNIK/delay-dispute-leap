# -*- coding: utf-8 -*-
"""Utilities for reproducibility manifests and forensic audit outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


def relpath_str(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_requirements(requirements_path: Path) -> List[str]:
    pkgs: List[str] = []
    if not requirements_path.exists():
        return pkgs
    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        pkg = re.split(r"[<>=!~]", line, maxsplit=1)[0].strip()
        if pkg:
            pkgs.append(pkg)
    return pkgs


def collect_package_versions(requirements_path: Path) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for pkg in parse_requirements(requirements_path):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def build_artifact_hashes(paths: Sequence[Path], project_root: Path) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for path in paths:
        out[relpath_str(path, project_root)] = sha256_file(path)
    return out


def build_run_manifest(
    project_root: Path,
    requirements_path: Path,
    artifact_paths: Sequence[Path],
    *,
    model_name: Optional[str],
    prompt_template_version: Optional[str],
    embedding_model: Optional[str],
    label_schema_version: Optional[str],
    command: Optional[str],
    seed: Optional[int],
    split_mode: Optional[str],
    text_mode: Optional[str],
    train_label_file: Optional[Path],
    eval_label_file: Optional[Path],
    metric_source_files: Sequence[Path],
    audit_status: str = "complete",
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    manifest = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": collect_package_versions(requirements_path),
        "model_name": model_name,
        "prompt_template_version": prompt_template_version,
        "embedding_model": embedding_model,
        "label_schema_version": label_schema_version,
        "command": command,
        "seed": seed,
        "split_mode": split_mode,
        "text_mode": text_mode,
        "train_label_file": relpath_str(train_label_file, project_root) if train_label_file else None,
        "eval_label_file": relpath_str(eval_label_file, project_root) if eval_label_file else None,
        "metric_source_files": [relpath_str(p, project_root) for p in metric_source_files if p],
        "artifact_hashes": build_artifact_hashes(artifact_paths, project_root),
        "audit_status": audit_status,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: Path, manifest: Dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def previous_final_eval_run(results_root: Path, current_run: Path) -> Optional[Path]:
    runs = sorted([p for p in results_root.glob("final_eval_*") if p.is_dir() and p.resolve() != current_run.resolve()])
    return runs[-1] if runs else None


def synthetic_file_diff(old_paths: Sequence[Path], new_paths: Sequence[Path], project_root: Path, comparison_id: str) -> pd.DataFrame:
    old_map = {relpath_str(p, project_root): p for p in old_paths}
    new_map = {relpath_str(p, project_root): p for p in new_paths}
    rows = []
    for rel in sorted(set(old_map) | set(new_map)):
        old_p = old_map.get(rel)
        new_p = new_map.get(rel)
        old_hash = sha256_file(old_p) if old_p else None
        new_hash = sha256_file(new_p) if new_p else None
        rows.append({
            "comparison_id": comparison_id,
            "path": rel,
            "exists_in_old": int(old_p is not None and old_p.exists()),
            "exists_in_new": int(new_p is not None and new_p.exists()),
            "sha256_old": old_hash,
            "sha256_new": new_hash,
            "content_changed": int(old_hash != new_hash),
        })
    return pd.DataFrame(rows)


def write_synthetic_git_summary(path: Path, note: str) -> None:
    path.write_text(note.strip() + "\n", encoding="utf-8")


def parse_holdout_report(report_path: Path) -> Dict[str, Optional[float]]:
    out = {"summary_accuracy": None, "summary_macro_f1": None, "summary_weighted_f1": None}
    if not report_path.exists():
        return out
    text = report_path.read_text(encoding="utf-8", errors="ignore")
    m_acc = re.search(r"accuracy\s+([0-9.]+)\s+\d+", text)
    m_macro = re.search(r"macro avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+\d+", text)
    m_weight = re.search(r"weighted avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+\d+", text)
    if m_acc:
        out["summary_accuracy"] = float(m_acc.group(1))
    if m_macro:
        out["summary_macro_f1"] = float(m_macro.group(1))
    if m_weight:
        out["summary_weighted_f1"] = float(m_weight.group(1))
    return out
