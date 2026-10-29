"""Optional helpers for testing LLM requests with VCR.py or pytest-recording.

VCR.py is only needed when using these helpers, not to import this module.
"""

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vcr import VCR
    from vcr.request import Request


def _json_equal(left: Any, right: Any) -> bool:
    # JSON booleans are not numbers, unlike Python's bool subclass of int.
    # JSON numbers compare by value, so 1 and 1.0 intentionally match.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def is_json_body_equal(r1: "Request", r2: "Request") -> None:
    """Match complete JSON values, or exact raw bodies for non-JSON requests."""
    try:
        body1 = json.loads(r1.body)
        body2 = json.loads(r2.body)
    except (TypeError, ValueError):
        assert r1.body == r2.body, "Request body does not match"
    else:
        assert _json_equal(body1, body2), "JSON body does not match"


def register_vcr_matchers(vcr: "VCR") -> None:
    """Register the JSON body matcher used by :func:`get_vcr_config`."""
    vcr.register_matcher("is_json_body_equal", is_json_body_equal)


def _before_record_response(response: dict[str, Any]) -> dict[str, Any]:
    # HTTP header names are case-insensitive; recorded response dictionaries
    # do not necessarily provide case-insensitive lookups.
    filtered_headers = {
        "authorization",
        "openai-organization",
        "openai-project",
        "x-api-key",
        "cookie",
        "set-cookie",
    }
    response["headers"] = {
        key: value
        for key, value in response["headers"].items()
        if key.lower() not in filtered_headers
    }
    return response


def get_vcr_config() -> dict[str, Any]:
    """Return fresh VCR settings for body matching and common credential headers.

    Settings apply to all requests recorded by the configured VCR instance.
    This is not a general secret scrubber: review cassettes before sharing.
    """
    return {
        "match_on": [
            "method",
            "scheme",
            "host",
            "port",
            "path",
            "query",
            "is_json_body_equal",
        ],
        "filter_headers": [
            "authorization",
            "openai-organization",
            "openai-project",
            "x-api-key",
            "cookie",
            "set-cookie",
        ],
        "before_record_response": _before_record_response,
    }
