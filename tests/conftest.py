from typing import Any

import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True, scope="session")
def _load_dotenv():
    # Load .env file so API keys are available when rewriting VCR cassettes
    load_dotenv()


@pytest.fixture(autouse=True, scope="session")
def vcr_config():
    def before_record_response(response: dict[str, Any]):
        filter_response_headers = [
            "openai-organization",
            "Set-Cookie",
        ]
        for response_header in filter_response_headers:
            response["headers"].pop(response_header, None)
        return response

    return {
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
    for item in items:
        # Apply vcr marker to all LLM tests
        llm_markers = [
            "anthropic",
            "litellm_anthropic",
            "litellm_openai",
            "mistral",
            "openai",
        ]
        if any(marker in item.keywords for marker in llm_markers):
            item.add_marker(pytest.mark.vcr)
