"""Tests for PromptFunction."""

from agentic.prompt_function import prompt


def test_decorator_return_str():
    @prompt()
    def get_capital(country: str) -> str:
        """What is the capital of {country}? Name only. No punctuation."""

    assert get_capital("Ireland") == "Dublin"


def test_decorator_return_bool():
    @prompt()
    def is_capital(capital: str, country: str) -> bool:
        """True if {capital} is the capital of {country}."""

    assert is_capital("Dublin", "Ireland") is True
