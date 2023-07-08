from enum import Enum
from typing import Generic, TypeVar

from agentic.function_call import FunctionCall

T = TypeVar("T")


class MessageRole(Enum):
    ASSISTANT = "assistant"
    FUNCTION = "function"
    SYSTEM = "system"
    USER = "user"


class Message:
    def __init__(self, role: MessageRole, content: str):
        self.role = role
        self.content = content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.role}, {self.content!r})"

    def value(self) -> str:
        return self.content


class UserMessage(Message):
    role = MessageRole.USER

    def __init__(self, content: str):
        self.content = content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.content!r})"

    @property
    def value(self) -> str:
        return self.content


class AssistantMessage(Message, Generic[T]):
    role = MessageRole.ASSISTANT

    def __init__(self, content: T):
        self.content = content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.content!r})"

    @property
    def value(self) -> T:
        return self.content


class FunctionResultMessage(Message, Generic[T]):
    role = MessageRole.FUNCTION

    def __init__(self, function_name: str, function_result: T):
        self.function_name = function_name
        self.function_result = function_result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.function_name!r}, {self.function_result!r})"

    @property
    def value(self) -> T:
        return self.function_result


class FunctionCallMessage(Message, Generic[T]):
    role = MessageRole.ASSISTANT

    def __init__(self, function_name: str, function_call: FunctionCall[T]):
        self.function_name = function_name
        self.function_call = function_call

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.function_name!r}, {self.function_call!r})"
        )

    @property
    def value(self) -> FunctionCall[T]:
        return self.function_call

    def get_result_message(self) -> FunctionResultMessage[T]:
        return FunctionResultMessage(
            function_name=self.function_name,
            function_result=self.function_call(),
        )
