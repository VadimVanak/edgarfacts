# src/edgarfacts/logging_utils.py
"""
Logging utilities for edgarfacts.

We intentionally keep logging lightweight and dependency-free.
The package exposes `get_logger()` as part of the public API.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, TextIO, Union


def get_logger(
    name: str = "edgarfacts",
    level: Union[int, str] = logging.INFO,
    stream: Optional[TextIO] = None,
    fmt: str = "%(message)s",
    datefmt: Optional[str] = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Configure and return a logger suitable for both notebooks and production use.

    - Adds exactly one StreamHandler (no duplicates across repeated calls).
    - Uses a minimal default format (message only).
    - Defaults to stdout to play nicely with notebook environments and many job runners.

    Parameters
    ----------
    name:
        Logger name. Defaults to "edgarfacts".
    level:
        Logging level (e.g., logging.INFO or "INFO").
    stream:
        Stream to log to. Defaults to sys.stdout.
    fmt:
        Logging format. Defaults to "%(message)s".
    datefmt:
        Optional date format for the formatter.
    propagate:
        Whether to propagate logs to ancestor loggers. Defaults to False to avoid
        duplicate output when the root logger is configured elsewhere.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Resolve stream early
    if stream is None:
        stream = sys.stdout

    # Normalize string level to int
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.INFO

    logger.setLevel(level)
    logger.propagate = propagate

    # Avoid adding duplicate stream handlers to the same stream/formatter.
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is stream:
            # Keep handler formatting consistent with the requested formatter.
            h.setLevel(level)
            h.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            return logger

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(handler)

    return logger
