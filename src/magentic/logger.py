import logging
import sys
from functools import cache

from magentic.settings import get_settings

logger = logging.getLogger("magentic")

# Set default log level to WARNING so magentic INFO logs must be explicitly enabled
if logger.level == logging.NOTSET:
    logger.setLevel(logging.WARNING)


@cache  # Can only be called once
def set_verbose():
    """Set the magentic logger to print INFO level messages to stdout."""
    logger.addHandler(logging.StreamHandler(sys.stdout))
    if logger.level > logging.INFO:
        logger.setLevel(logging.INFO)


settings = get_settings()
if settings.verbose:
    set_verbose()
