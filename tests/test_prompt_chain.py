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
