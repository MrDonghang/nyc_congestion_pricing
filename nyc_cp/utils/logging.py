"""Logging setup shared across CLI scripts."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(name: str, log_root: str | Path = "logs", level: int = logging.INFO) -> Path:
    """Configure root logger with timestamped file + console handlers.

    Returns the log-file path.
    """
    log_dir = Path(log_root) / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [fh, sh]

    # Quiet noisy libs.
    for noisy in ("cmdstanpy", "prophet", "stan", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    os.environ.setdefault("STAN_BACKEND", "CMDSTANPY")
    os.environ.setdefault("CMDSTANPY_LOGGING_LEVEL", "ERROR")

    return log_file
