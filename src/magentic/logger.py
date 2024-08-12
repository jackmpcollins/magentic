import logging

import logfire_api

logger = logging.getLogger("magentic")
# Set default log level to WARNING so INFO logs must be explicitly enabled
if logger.level == logging.NOTSET:
    logger.setLevel(logging.WARNING)

logfire = logfire_api.Logfire(otel_scope="magentic")  # TODO: Pass version here too
