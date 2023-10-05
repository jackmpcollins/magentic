import pytest

from magentic.prompt_chain import prompt_chain, MaxFunctionCallsError
from unittest.mock import Mock
from magentic.chat_model.message import AssistantMessage

@pytest.mark.openai
def test_prompt_chain():
    def get_current_weather(location, unit="fahrenheit"):
        """Get the current weather in a given location"""
        return {
            "location": location,
            "temperature": "72",
            "unit": unit,
            "forecast": ["sunny", "windy"],
        }

    @prompt_chain(
        template="What's the weather like in {city}?",
        functions=[get_current_weather],
    )
    def describe_weather(city: str) -> str:
        ...

    output = describe_weather("Boston")
    assert isinstance(output, str)

def test_prompt_chain_max_calls():
    def get_current_weather(location, unit="fahrenheit"):
        """Get the current weather in a given location"""
        return {
            "location": location,
            "temperature": "72",
            "unit": unit,
            "forecast": ["sunny", "windy"],
        }
    mock_model = Mock()
    mock_model.complete.return_value = AssistantMessage(content="Weather yay!") 
    @prompt_chain(
        template="What's the weather like in {city}?",
        functions=[get_current_weather],
        model=mock_model,
        max_calls=0
    )
    def describe_weather(city: str) -> str:
        ...

    with pytest.raises(MaxFunctionCallsError):
        describe_weather("Boston")


@pytest.mark.asyncio
@pytest.mark.openai
async def test_async_prompt_chain():
    async def get_current_weather(location, unit="fahrenheit"):
        """Get the current weather in a given location"""
        return {
            "location": location,
            "temperature": "72",
            "unit": unit,
            "forecast": ["sunny", "windy"],
        }

    @prompt_chain(
        template="What's the weather like in {city}?",
        functions=[get_current_weather],
    )
    async def describe_weather(city: str) -> str:
        ...

    output = await describe_weather("Boston")
    assert isinstance(output, str)


