import logging
import sys

from magentic.settings import get_settings

settings = get_settings()

logger = logging.getLogger("magentic")

if settings.verbose:
    logger.addHandler(logging.StreamHandler(sys.stdout))

if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO if settings.verbose else logging.WARNING)
