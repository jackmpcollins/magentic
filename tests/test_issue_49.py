import re
from typing import Generic, TypeVar

import pytest
from pydantic import BaseModel

from magentic import prompt_chain
from magentic.chat_model.openai_chat_model import (
    OpenaiChatCompletionChoiceMessage,
    OpenaiMessageRole,
)

T = TypeVar('T')


class ApiResponse(BaseModel, Generic[T]):
    error: str | None = None
    response: T


def test_promptchain_with_generic():
    def get_current_weather() -> ApiResponse[str]:
        """Get the current weather"""
        return ApiResponse[str](response="sunny")

    @prompt_chain(
        template="What's the weather like ?",
        functions=[get_current_weather]
    )
    def describe_weather() -> str:
        ...

    output = describe_weather()
    assert isinstance(output, str)


@pytest.mark.parametrize('name', (
        "foo",
        "x" * 64,
        "return_foo[str]",
))
def test_openai_chat_completion_choice_message_name_is_cleaned_up(name: str):
    openai_message = OpenaiChatCompletionChoiceMessage(
        role=OpenaiMessageRole.USER,
        content="foo",
        name=name,
    )
    assert openai_message.name is not None
    assert re.match(pattern=r'^[a-zA-Z0-9_-]{1,64}$', string=openai_message.name)
