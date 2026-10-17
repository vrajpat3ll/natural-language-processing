import os
import logging
from datetime import datetime
from pathlib import Path

# generate the log file for a run
root_dir = Path(__file__).resolve().parents[0]
log_dir = root_dir / "logs"
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(level=logging.INFO)


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger instance for the given module name.
    Logs to both console and a single run-specific file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Formatter
    _old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = _old_factory(*args, **kwargs)
        try:
            record.name = os.path.relpath(record.pathname, start=str(root_dir))
        except Exception:
            record.name = record.pathname
        return record

    logging.setLogRecordFactory(record_factory)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (same file for the whole run)
    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Prevent double logging via root logger
    logger.propagate = False

    return logger
