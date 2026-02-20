import logging
import os
import sys
from datetime import datetime

def setup_logger(run_id: str, log_level=logging.DEBUG) -> logging.Logger:
    """
    Configures a DUAL LOGGING SYSTEM for XAI:
    - Standard Log (INFO): inference_run_{run_id}.log
    - Extended Log (DEBUG): inference_run_{run_id}_extended.log

    The logger itself is set to DEBUG level to capture all messages.
    Individual handlers filter messages based on their level.

    Args:
        run_id (str): Unique identifier for the experiment run.
        log_level (int): Base logging level (default: logging.DEBUG).

    Returns:
        logging.Logger: Configured logger with dual file handlers + console.
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Log filenames
    standard_log_file = os.path.join(log_dir, f"inference_{run_id}.log")
    extended_log_file = os.path.join(log_dir, f"inference_{run_id}_extended.log")

    # Create logger
    logger = logging.getLogger(f"experiment_{run_id}")
    logger.setLevel(logging.DEBUG)  # Capture all messages at logger level

    # Clear existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # ========== HANDLER 1: Standard Log (INFO level) ==========
    standard_handler = logging.FileHandler(standard_log_file, mode='w', encoding='utf-8')
    standard_handler.setLevel(logging.INFO)  # Only INFO and above
    standard_handler.setFormatter(formatter)
    logger.addHandler(standard_handler)

    # ========== HANDLER 2: Extended Log (DEBUG level) ==========
    extended_handler = logging.FileHandler(extended_log_file, mode='w', encoding='utf-8')
    extended_handler.setLevel(logging.DEBUG)  # Capture DEBUG and above
    extended_handler.setFormatter(formatter)
    logger.addHandler(extended_handler)

    # ========== HANDLER 3: Console (INFO level) ==========
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Only show INFO+ on console
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Dual Logger initialized. Standard: {standard_log_file}, Extended: {extended_log_file}")
    return logger
