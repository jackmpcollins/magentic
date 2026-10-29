# Testing with VCR.py

[VCR.py](https://vcrpy.readthedocs.io/) records HTTP requests and responses in cassette files and replays them in later test runs. This lets you test LLM calls without contacting the provider on every run.

Install the optional testing tools separately. Magentic does not install or import VCR.py for normal use.

```bash
pip install pytest pytest-recording vcrpy
```

## pytest-recording setup

Add the helpers to your `conftest.py`:

```python
import pytest
from magentic.testing import get_vcr_config, register_vcr_matchers


def pytest_recording_configure(config, vcr):
    register_vcr_matchers(vcr)


@pytest.fixture(scope="session")
def vcr_config():
    return get_vcr_config()
```

Mark tests that call an LLM with `@pytest.mark.vcr`. Record deliberately with your provider credentials available, then replay without credentials using `pytest --record-mode=none`. Replay does not require a real API key, but your model client may still require a dummy key to initialize. Initial recording calls the provider and can incur charges.

The configuration matches the method, scheme, host, port, path, query and complete JSON body. JSON object key order and whitespace do not matter; message contents and list order do. JSON booleans do not match numbers (`true` differs from `1`), including in nested objects and lists. Numbers compare by value, so `1` and `1.0` match. Non-JSON bodies must match exactly. This avoids replaying a response recorded for a different prompt.

## Using VCR.py directly

```python
import vcr
from magentic.testing import get_vcr_config, register_vcr_matchers

recorder = vcr.VCR(**get_vcr_config())
register_vcr_matchers(recorder)

with recorder.use_cassette("tests/cassettes/example.yaml", record_mode="none"):
    # Call the function under test here.
    pass
```

## Review cassettes before sharing

The helpers remove these request and response headers, regardless of letter case: `Authorization`, `X-Api-Key`, `OpenAI-Organization`, `OpenAI-Project`, `Cookie` and `Set-Cookie`. They apply to every request handled by this VCR configuration, including custom provider URLs. Use a separate VCR instance when you do not want these settings on other services.

This is not a general secret scrubber. Prompts, responses, URLs, query parameters, request bodies and other headers can still contain private information or credentials. Review every cassette before committing or sharing it. Add project-specific filters when needed. If you extend the configuration, preserve the existing filters and matcher registration.
