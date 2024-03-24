import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True, scope="session")
def _load_dotenv():
    load_dotenv()
