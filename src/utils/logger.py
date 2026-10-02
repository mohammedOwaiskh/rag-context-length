import logging
from datetime import datetime

from utils import get_project_root

DEBUG = logging.DEBUG
INFO = logging.INFO
ERROR = logging.ERROR


def setup_logger(name: str, log_level=logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance with both file and console handlers.

    Reuses existing handlers if this logger has already been configured.

    Args:
        name: Logger name, typically the calling module's name.
        log_level: Minimum level emitted by the handlers.

    Returns:
        The configured logger.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    # Create logs directory if it doesn't exist
    logs_dir = get_project_root() / "logs" / name
    logs_dir.mkdir(parents=True,exist_ok=True)

    # Log file name with timestamp
    log_file = logs_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger