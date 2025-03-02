import json
import os
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from dotenv import load_dotenv
from pytest_mock import MockerFixture
from vcr import VCR
from vcr.request import Request


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
    """Register VCR matcher for JSON request bodies"""

    def is_json_body_equal(r1: Request, r2: Request) -> None:
        try:
            msg = "JSON body does not match"
            assert json.loads(r1.body) == json.loads(r2.body), msg  # type: ignore[arg-type, unused-ignore]
        except (TypeError, json.JSONDecodeError):
            return

    vcr.register_matcher("is_json_body_equal", is_json_body_equal)


@pytest.fixture(autouse=True, scope="session")
def vcr_config():
    """Configure VCR to match on JSON bodies and filter sensitive headers"""

    def before_record_response(response: dict[str, Any]) -> dict[str, Any]:
        filter_response_headers = [
            "openai-organization",
            "Set-Cookie",
        ]
        for response_header in filter_response_headers:
            response["headers"].pop(response_header, None)
        return response

    return {
        "match_on": ["method", "host", "path", "query", "is_json_body_equal"],
        "filter_headers": [
            # openai
            "authorization",
            "openai-organization",
            # anthropic
            "x-api-key",
            # litellm_openai
            "cookie",
        ],
        "before_record_response": before_record_response,
    }


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
        "openai_ollama",
        "openai_xai",
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
