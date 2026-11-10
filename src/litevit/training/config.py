"""Training config loading and path resolution."""

from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_KEYS = ("model", "data", "epochs", "imgsz", "batch")
ALLOWED_ARCHS = ("baseline", "v1_p5", "gate_only", "gate_p5", "gate_edge_cbam")
DEFAULT_PROJECT = "runs/train"


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{path} missing required keys: {', '.join(missing)}")
    cls_pw = float(payload.get("cls_pw", 0.0))
    if not 0.0 <= cls_pw <= 1.0:
        raise ValueError(f"cls_pw must be in [0, 1], got {cls_pw}")
    arch = str(payload.get("arch", "baseline"))
    if arch not in ALLOWED_ARCHS:
        raise ValueError(f"arch must be one of {ALLOWED_ARCHS}, got {arch}")
    return payload


def resolve_paths(config: dict, root: Path) -> dict:
    resolved = dict(config)
    data_path = Path(resolved["data"])
    if not data_path.is_absolute():
        resolved["data"] = str((root / data_path).resolve())
    project = Path(resolved.get("project", DEFAULT_PROJECT))
    if not project.is_absolute():
        project = (root / project).resolve()
    resolved["project"] = str(project)
    model = Path(resolved["model"])
    if model.suffix in {".yaml", ".yml"} or model.suffix == ".pt":
        candidate = model if model.is_absolute() else root / model
        if candidate.exists():
            resolved["model"] = str(candidate.resolve())
    return resolved
