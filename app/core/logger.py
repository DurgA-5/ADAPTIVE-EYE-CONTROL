"""AdaptiveEye Structured Logging System."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """Configure the root logger for AdaptiveEye with standard formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write log records to disk.
        log_format: Custom log format string.
        date_format: Custom timestamp format string.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = log_format or "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    dfmt = date_format or "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(file_path), encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt=dfmt,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a named logger prefixed with 'AdaptiveEye'.
    
    Args:
        name: Sub-module or component name.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger_name = f"AdaptiveEye.{name}" if not name.startswith("AdaptiveEye") else name
    return logging.getLogger(logger_name)
