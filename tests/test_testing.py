import os
import subprocess
import sys
from pathlib import Path

import pytest
from vcr import VCR
from vcr.request import Request

from magentic.testing import get_vcr_config, is_json_body_equal, register_vcr_matchers


def request(body: bytes | None, headers: dict[str, str] | None = None) -> Request:
    return Request("POST", "https://example.invalid/v1/chat", body, headers or {})


def test_json_body_matching():
    is_json_body_equal(
        request(b'{"messages": ["hello"], "model":"m"}'),
        request(b'{"model":"m","messages":["hello"]}'),
    )
    with pytest.raises(AssertionError, match="JSON body"):
        is_json_body_equal(
            request(b'{"messages":["hello"]}'), request(b'{"messages":["bye"]}')
        )
    with pytest.raises(AssertionError, match="JSON body"):
        is_json_body_equal(request(b"[1,2]"), request(b"[2,1]"))


@pytest.mark.parametrize(
    ("left", "right"), [(b"hello", b"bye"), (b"{}", b"not-json"), (None, b"null")]
)
def test_non_json_must_match_exactly(left, right):
    with pytest.raises(AssertionError, match="Request body"):
        is_json_body_equal(request(left), request(right))


@pytest.mark.parametrize("body", [None, b"raw-data", b""])
def test_equal_non_json(body):
    is_json_body_equal(request(body), request(body))


def test_config_fresh():
    first = get_vcr_config()
    first["filter_headers"].clear()
    first["match_on"].clear()
    assert get_vcr_config()["filter_headers"]
    assert "is_json_body_equal" in get_vcr_config()["match_on"]


@pytest.mark.parametrize(
    "name",
    [
        "Authorization",
        "AUTHORIZATION",
        "X-API-KEY",
        "x-api-key",
        "OpenAI-Organization",
        "OPENAI-PROJECT",
        "Cookie",
        "Set-Cookie",
    ],
)
def test_credential_header_filtering_roundtrip(tmp_path, name):
    recorder = VCR(**get_vcr_config())
    register_vcr_matchers(recorder)
    cassette_path = str(tmp_path / "headers.yaml")
    # Synthetic credentials only. No sockets, HTTP clients or real API keys.
    with recorder.use_cassette(cassette_path) as cassette:
        cassette.append(
            request(
                b'{"messages":["hello"]}',
                {
                    name: "placeholder-request-secret",
                    "Content-Type": "application/json",
                },
            ),
            {
                "status": {"code": 200, "message": "OK"},
                "headers": {
                    name: ["placeholder-response-secret"],
                    "Content-Type": ["application/json"],
                },
                "body": {"string": b"{}"},
            },
        )
    text = Path(cassette_path).read_text()
    assert "placeholder-request-secret" not in text
    assert "placeholder-response-secret" not in text
    with recorder.use_cassette(
        cassette_path, record_mode="none", allow_playback_repeats=True
    ) as cassette:
        match = cassette.play_response(request(b'{"messages": ["hello"]}'))
        assert match["headers"]["Content-Type"] == ["application/json"]
        with pytest.raises(KeyError, match="doesn.t contain the request"):
            cassette.play_response(request(b'{"messages":["different"]}'))


def test_import_without_vcr():
    code = """
import importlib.abc
import sys
class BlockVCR(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vcr' or fullname.startswith('vcr.'):
            raise ImportError('VCR deliberately unavailable')
sys.meta_path.insert(0, BlockVCR())
from magentic.testing import get_vcr_config
assert get_vcr_config()['filter_headers']
assert 'vcr' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (b"true", b"1"),
        (b"false", b"0"),
        (b'{"nested":[{"flag":true}]}', b'{"nested":[{"flag":1}]}'),
        (b'{"nested":[false]}', b'{"nested":[0]}'),
    ],
)
def test_json_boolean_is_not_number(left, right):
    with pytest.raises(AssertionError, match="JSON body"):
        is_json_body_equal(request(left), request(right))
    with pytest.raises(AssertionError, match="JSON body"):
        is_json_body_equal(request(right), request(left))


def test_json_numbers_compare_by_value():
    is_json_body_equal(request(b'{"value":[1,0]}'), request(b'{"value":[1.0,0.0]}'))


def test_cassette_rejects_boolean_number_substitution(tmp_path):
    recorder = VCR(**get_vcr_config())
    register_vcr_matchers(recorder)
    path = str(tmp_path / "boolean.yaml")
    with recorder.use_cassette(path) as cassette:
        cassette.append(
            request(b'{"nested":[{"flag":true}]}'),
            {
                "status": {"code": 200, "message": "OK"},
                "headers": {},
                "body": {"string": b"correct"},
            },
        )
    with recorder.use_cassette(
        path, record_mode="none", allow_playback_repeats=True
    ) as cassette:
        assert (
            cassette.play_response(request(b'{"nested": [{"flag": true}]}'))["body"][
                "string"
            ]
            == b"correct"
        )
        with pytest.raises(KeyError, match="doesn.t contain the request"):
            cassette.play_response(request(b'{"nested":[{"flag":1}]}'))
