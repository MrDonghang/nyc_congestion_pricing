"""Config loading and path resolution.

A run is described by three configs:
  - paths.yaml         (where data/output live)
  - modes/<mode>.yaml  (per-mode preprocessing knobs + forecast windows)
  - models/<name>.yaml (per-model hyperparameters)

Forecast windows are per-mode because citibike and replica do not share the
year-displaced validation window that bus/subway use.

Local overrides go in configs/paths.local.yaml (gitignored).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

Direction = Literal["all", "O", "D"]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_paths() -> dict[str, Any]:
    base = _read_yaml(CONFIG_DIR / "paths.yaml")
    local = CONFIG_DIR / "paths.local.yaml"
    if local.exists():
        base = _deep_merge(base, _read_yaml(local))
    return base


def load_mode(mode: str) -> dict[str, Any]:
    return _read_yaml(CONFIG_DIR / "modes" / f"{mode}.yaml")


def load_model(name: str) -> dict[str, Any]:
    return _read_yaml(CONFIG_DIR / "models" / f"{name}.yaml")


@dataclass(frozen=True)
class Window:
    train_end: str
    test_start: str
    test_end: str

    @property
    def name_suffix(self) -> str:
        return f"{self.train_end}_to_{self.test_end}"


def get_window(mode: str, name: Literal["validation", "test"], mode_cfg: dict | None = None) -> Window:
    mode_cfg = mode_cfg or load_mode(mode)
    w = mode_cfg["windows"][name]
    return Window(train_end=w["train_end"], test_start=w["test_start"], test_end=w["test_end"])


# ----------------------------------------------------------------------
# Path resolution
# ----------------------------------------------------------------------

def _resolve(root: str | Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


def actual_csv(mode: str, direction: Direction = "all", paths: dict | None = None) -> Path:
    """Path to the daily/weekly ridership CSV for ``mode``."""
    paths = paths or load_paths()
    spec = paths["modes"][mode]
    direction_long = {"O": "origin", "D": "destination"}.get(direction, "all")
    template: str = spec["actual"]
    rel = template.format(direction=direction, direction_long=direction_long)
    return _resolve(paths["data_root"], rel)


def output_dir(mode: str, model: str, direction: Direction = "all", paths: dict | None = None) -> Path:
    """Where forecasts for ``(mode, model, direction)`` are saved."""
    paths = paths or load_paths()
    base = Path(paths["output_root"]) / mode / model
    if direction != "all":
        base = base / direction
    base.mkdir(parents=True, exist_ok=True)
    return base


def log_dir(name: str, paths: dict | None = None) -> Path:
    paths = paths or load_paths()
    p = Path(paths["log_root"]) / name
    p.mkdir(parents=True, exist_ok=True)
    return p
