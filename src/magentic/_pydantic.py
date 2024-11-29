from collections.abc import Callable
from typing import Any, TypeVar

import openai
from pydantic import BaseModel
from pydantic import ConfigDict as _ConfigDict
from pydantic import with_config as _with_config


class ConfigDict(_ConfigDict, total=False):
    openai_strict: bool


T = TypeVar("T", bound=type | Callable[..., Any])


# Relax type constraints to allow as function decorator
def with_config(config: ConfigDict) -> Callable[[T], T]:
    return _with_config(config)  # type: ignore[return-value]


def get_pydantic_config(obj: Any) -> ConfigDict | None:
    return getattr(obj, "__pydantic_config__", None)


# TODO: Use directly from pydantic when available
# https://github.com/pydantic/pydantic/issues/5213
def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generates a JSON schema for a pydantic model.

    If `openai_strict` is set to `True` in the model's config, a schema compatible with
    OpenAI APIs strict mode is generated.
    """
    if model.model_config.get("openai_strict", False):
        tool_param = openai.pydantic_function_tool(model)
        return tool_param["function"].get("parameters", {})
    model_schema = model.model_json_schema().copy()
    model_schema.pop("title", None)
    model_schema.pop("description", None)
    return model_schema
