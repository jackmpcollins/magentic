import pytest

from magentic.prompt_chain import prompt_chain


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


@pytest.mark.openai
def test_prompt_chain_max_calls():
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
        max_calls=0
    )
    def describe_weather(city: str) -> str:
        ...

    with pytest.raises(PermissionError):
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


