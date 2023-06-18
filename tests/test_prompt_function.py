"""Tests for PromptFunction."""

from agentic.prompt_function import prompt


def test_decorator_return_str():
    @prompt()
    def get_capital(country: str) -> str:
        """What is the capital of {country}? Name only. No punctuation."""

    assert get_capital("Ireland") == "Dublin"
