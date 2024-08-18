from magentic._pydantic import ConfigDict as ConfigDict
from magentic.chat_model.message import (
    AnyMessage,
    AssistantMessage,
    FunctionResultMessage,
    Placeholder,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic.chatprompt import chatprompt
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)
from magentic.prompt_chain import prompt_chain
from magentic.prompt_function import prompt
from magentic.streaming import AsyncStreamedStr, StreamedStr

__all__ = [
    "AnyMessage",
    "AssistantMessage",
    "FunctionResultMessage",
    "Placeholder",
    "SystemMessage",
    "ToolResultMessage",
    "UserMessage",
    "OpenaiChatModel",
    "chatprompt",
    "AsyncParallelFunctionCall",
    "FunctionCall",
    "ParallelFunctionCall",
    "prompt_chain",
    "prompt",
    "AsyncStreamedStr",
    "StreamedStr",
]
