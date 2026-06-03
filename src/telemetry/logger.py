import os
import logging

# Resolve the absolute path of the workspace root to place log files in data/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LOG_PATH = os.path.join(ROOT_DIR, "data", "app.log")

def setup_logger(name: str = "agentic_ai", log_path: str = DEFAULT_LOG_PATH) -> logging.Logger:
    """Configures and returns a standard logger that logs to stdout and appends to data/app.log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if logger is re-initialized
    if logger.handlers:
        return logger

    # Format defining: timestamp, level, file/line metadata, and message
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console output handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Persistent log file handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()
