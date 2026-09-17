"""Logging utility for the customer support agent project.

Provides standardized formatting, timestamps, log levels, and console/file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "support_agent", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger instance.

    Args:
        name: Name of the logger, typically __name__.
        log_file: Optional path to append log records.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.hasHandlers():
        return logger

    # Formatter with ISO-like timestamp, level, module name and message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console / stdout handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
