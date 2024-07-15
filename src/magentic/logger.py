import logging

logger = logging.getLogger("magentic")
# Set default log level to WARNING so INFO logs must be explicitly enabled
if logger.level == logging.NOTSET:
    logger.setLevel(logging.WARNING)
