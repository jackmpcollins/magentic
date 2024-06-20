import logging

# Set log level to WARNING so logs are only created when explicitly enabled
logger = logging.getLogger("magentic")
if logger.level == logging.NOTSET:
    logger.setLevel(logging.WARNING)
