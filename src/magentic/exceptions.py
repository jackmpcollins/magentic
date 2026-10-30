"""Public exceptions raised by Magentic.

These are aliases of the existing classes, so existing import paths still work.
"""

from magentic.chat_model.base import (
    FunctionCallNotAllowedError as FunctionCallNotAllowedError,
)
from magentic.chat_model.base import ObjectNotAllowedError as ObjectNotAllowedError
from magentic.chat_model.base import StringNotAllowedError as StringNotAllowedError
from magentic.chat_model.base import ToolSchemaParseError as ToolSchemaParseError
from magentic.chat_model.base import UnknownToolError as UnknownToolError
from magentic.prompt_chain import MaxFunctionCallsError as MaxFunctionCallsError
