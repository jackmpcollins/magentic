from typing import Annotated

import pytest
from pydantic import AfterValidator, BaseModel

from magentic.chat_model.anthropic_chat_model import AnthropicChatModel
from magentic.chat_model.message import UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel
from magentic.chat_model.retry_chat_model import RetryChatModel


@pytest.mark.openai
def test_retry_chat_model_complete_openai():
    def assert_is_ireland(v):
        if v != "Ireland":
            msg = "Country must be Ireland."
            raise ValueError(msg)
        return v

    class Country(BaseModel):
        name: Annotated[str, AfterValidator(assert_is_ireland)]

    chat_model = RetryChatModel(OpenaiChatModel("gpt-4o-mini"), max_retries=3)
    message = chat_model.complete(
        messages=[UserMessage("Return a country.")],
        output_types=[Country],
    )
    assert isinstance(message.content, Country)
    assert message.content.name == "Ireland"


@pytest.mark.asyncio
@pytest.mark.openai
async def test_retry_chat_model_acomplete_openai():
    def assert_is_ireland(v):
        if v != "Ireland":
            msg = "Country must be Ireland."
            raise ValueError(msg)
        return v

    class Country(BaseModel):
        name: Annotated[str, AfterValidator(assert_is_ireland)]

    chat_model = RetryChatModel(OpenaiChatModel("gpt-4o-mini"), max_retries=3)
    message = await chat_model.acomplete(
        messages=[UserMessage("Return a country.")],
        output_types=[Country],
    )
    assert isinstance(message.content, Country)
    assert message.content.name == "Ireland"


@pytest.mark.anthropic
def test_retry_chat_model_complete_anthropic():
    def assert_is_ireland(v):
        if v != "Ireland":
            msg = "Country must be Ireland."
            raise ValueError(msg)
        return v

    class Country(BaseModel):
        name: Annotated[str, AfterValidator(assert_is_ireland)]

    chat_model = RetryChatModel(
        AnthropicChatModel("claude-3-haiku-20240307"), max_retries=3
    )
    message = chat_model.complete(
        messages=[UserMessage("Return a country.")],
        output_types=[Country],
    )
    assert isinstance(message.content, Country)
    assert message.content.name == "Ireland"


@pytest.mark.asyncio
@pytest.mark.anthropic
async def test_retry_chat_model_acomplete_anthropic():
    def assert_is_ireland(v):
        if v != "Ireland":
            msg = "Country must be Ireland."
            raise ValueError(msg)
        return v

    class Country(BaseModel):
        name: Annotated[str, AfterValidator(assert_is_ireland)]

    chat_model = RetryChatModel(
        AnthropicChatModel("claude-3-haiku-20240307"), max_retries=3
    )
    message = await chat_model.acomplete(
        messages=[UserMessage("Return a country.")],
        output_types=[Country],
    )
    assert isinstance(message.content, Country)
    assert message.content.name == "Ireland"
