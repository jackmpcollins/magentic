import logging

import logfire_api

logger = logging.getLogger("magentic")
# Set default log level to WARNING so INFO logs must be explicitly enabled
if logger.level == logging.NOTSET:
    logger.setLevel(logging.WARNING)

logfire = logfire_api.with_settings(
    custom_scope_suffix="magentic",  # TODO: This should be scope, not suffix
    otel_scope_version="",  # TODO: Not currently accepted
)
