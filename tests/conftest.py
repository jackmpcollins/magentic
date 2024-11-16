import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True, scope="session")
def _load_dotenv():
    load_dotenv()


@pytest.fixture(autouse=True, scope="session")
def vcr_config():
    def before_record_response(response):
        del response["headers"]["openai-organization"]
        del response["headers"]["Set-Cookie"]
        return response

    return {
        "filter_headers": ["authorization", "openai-organization"],
        "before_record_response": before_record_response,
    }
