from typing import Any, Callable, TypeVar

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
