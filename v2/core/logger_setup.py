
"""
Rotating Logger Setup
Ensures the system can run for 100 years without filling the disk.
Rotates log files when they reach 10MB, keeping only the last 5 backups.
"""
import logging
import logging.handlers
import os

def setup_logger(name="pluginfer", log_file="pluginfer.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # avoid duplicates
    if logger.handlers:
        return logger
        
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 1. Console Handler (Live View)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 2. Rotating File Handler (Disk Safety)
    # Max size 10MB, Keep 5 Backups
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    file_handler = logging.handlers.RotatingFileHandler(
        f"logs/{log_file}", maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
