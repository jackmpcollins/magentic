import os
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from uuid import UUID

import pytest
from dotenv import load_dotenv
from pytest_mock import MockerFixture
from vcr import VCR

from magentic.testing import get_vcr_config, register_vcr_matchers


@pytest.fixture(autouse=True, scope="session")
def _load_dotenv():
    """Load .env file so API keys are available when rewriting VCR cassettes"""
    # Prefer env vars from .env over existing env vars
    load_dotenv(override=True)
    # Set default values so tests can run using existing cassettes without env vars
    load_dotenv(".env.template")
    # Remove URL from error messages to avoid them changing with pydantic version
    os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "false"


def pytest_recording_configure(config: pytest.Config, vcr: VCR) -> None:
    register_vcr_matchers(vcr)


@pytest.fixture(autouse=True, scope="session")
def vcr_config():
    return get_vcr_config()


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    llm_markers = [
        "anthropic",
        "litellm_anthropic",
        "litellm_ollama",
        "litellm_openai",
        "mistral",
        "openai",
        "openai_gemini",
        "openai_ollama",
        "openai_xai",
        "openrouter",
    ]
    for item in items:
        # Apply vcr marker to all LLM tests
        if any(marker in item.keywords for marker in llm_markers):
            item.add_marker(pytest.mark.vcr)


@pytest.fixture(autouse=True)
def _mock_create_unique_id(mocker: MockerFixture) -> None:
    """Mock `uuid4` to make `_create_unique_id` return deterministic values"""

    def _mock_uuid4() -> Iterator[UUID]:
        for i in count():
            yield UUID(int=i)

    mocker.patch("magentic.function_call.uuid4", side_effect=_mock_uuid4())


def test_mock_create_unique_id():
    """Test that `_mock_create_unique_id` makes `_create_unique_id` deterministic"""
    from magentic.function_call import _create_unique_id

    assert _create_unique_id() == "000000000"
    assert _create_unique_id() == "000000001"


@pytest.fixture
def document_bytes_pdf() -> bytes:
    return Path("tests/data/test.pdf").read_bytes()


@pytest.fixture
def image_bytes_jpg() -> bytes:
    return Path("tests/data/python-powered.jpg").read_bytes()


@pytest.fixture
def image_bytes_png() -> bytes:
    return Path("tests/data/python-powered.png").read_bytes()
