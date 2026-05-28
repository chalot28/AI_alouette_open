"""
Logging — cấu hình logger thống nhất toàn project.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from config.config import config

# ──────────────────────────────────────────────
# Console
# ──────────────────────────────────────────────
console = Console()


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Lấy logger instance.

    Args:
        name: Tên logger (thường dùng __name__).
        level: Log level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Tránh duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level or config.logging.level)
    logger.propagate = False

    # Console handler (Rich)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        omit_repeated_times=True,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(level or config.logging.level)
    logger.addHandler(console_handler)

    # File handler
    log_dir = config.logging.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def get_rich_progress(iterable=None, desc: str = "") -> Progress:
    """
    Tạo Rich progress bar.

    Usage:
        for item in get_rich_progress(data_loader, desc="Training"):
            ...
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("/"),
        TimeRemainingColumn(),
        console=console,
    )
    if iterable is not None:
        return progress.track(iterable, description=desc)
    return progress
