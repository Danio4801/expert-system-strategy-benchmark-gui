import logging
import os
import sys
from datetime import datetime

def setup_logger(run_id: str, log_level=logging.DEBUG) -> logging.Logger:
















    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)


    standard_log_file = os.path.join(log_dir, f"inference_{run_id}.log")
    extended_log_file = os.path.join(log_dir, f"inference_{run_id}_extended.log")


    logger = logging.getLogger(f"experiment_{run_id}")
    logger.setLevel(logging.DEBUG)


    if logger.hasHandlers():
        logger.handlers.clear()


    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


    standard_handler = logging.FileHandler(standard_log_file, mode='w', encoding='utf-8')
    standard_handler.setLevel(logging.INFO)
    standard_handler.setFormatter(formatter)
    logger.addHandler(standard_handler)


    extended_handler = logging.FileHandler(extended_log_file, mode='w', encoding='utf-8')
    extended_handler.setLevel(logging.DEBUG)
    extended_handler.setFormatter(formatter)
    logger.addHandler(extended_handler)


    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Dual Logger initialized. Standard: {standard_log_file}, Extended: {extended_log_file}")
    return logger
