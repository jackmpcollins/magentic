"""Create chat models by provider name without importing optional adapters eagerly."""

import inspect
import warnings
from importlib import import_module
from typing import Any, Literal

from magentic.chat_model.base import ChatModel

_BUILTIN_MODELS = {
    "anthropic": ("magentic.chat_model.anthropic_chat_model", "AnthropicChatModel"),
    "litellm": ("magentic.chat_model.litellm_chat_model", "LitellmChatModel"),
    "mistral": ("magentic.chat_model.mistral_chat_model", "MistralChatModel"),
    "openai": ("magentic.chat_model.openai_chat_model", "OpenaiChatModel"),
    "openrouter": ("magentic.chat_model.openrouter_chat_model", "OpenRouterChatModel"),
}
_REGISTERED_MODELS: dict[str, type[ChatModel]] = {}


def register_chat_model(provider: str, chat_model: type[ChatModel]) -> None:
    """Register a provider whose class accepts a model name as its first argument."""
    if not provider or ":" in provider or any(c.isspace() for c in provider):
        msg = "Provider names must be nonempty and contain no colons or whitespace."
        raise ValueError(msg)
    if not isinstance(chat_model, type) or not issubclass(chat_model, ChatModel):
        msg = "chat_model must be a ChatModel subclass."
        raise TypeError(msg)
    if provider in _BUILTIN_MODELS or provider in _REGISTERED_MODELS:
        msg = f"Provider {provider!r} is already registered."
        raise ValueError(msg)
    _REGISTERED_MODELS[provider] = chat_model


def model(
    name: str,
    *,
    unsupported_args: Literal["raise", "warn", "ignore"] = "raise",
    **kwargs: Any,
) -> ChatModel:
    """Create a ChatModel from 'provider:model', passing options to its constructor.

    Only unknown keyword names are filtered by unsupported_args. Constructor errors
    and invalid values are never swallowed. Provider classes with **kwargs receive
    all supplied options and own their validation.
    """
    if unsupported_args not in ("raise", "warn", "ignore"):
        msg = "unsupported_args must be 'raise', 'warn', or 'ignore'."
        raise ValueError(msg)
    provider, separator, model_name = name.partition(":")
    if (
        not separator
        or not provider
        or not model_name
        or any(c.isspace() for c in provider)
    ):
        msg = "Model name must have the form 'provider:model' with nonempty parts."
        raise ValueError(msg)
    chat_model = _REGISTERED_MODELS.get(provider)
    if chat_model is None:
        if provider not in _BUILTIN_MODELS:
            msg = f"Unknown model provider {provider!r}. Register it with register_chat_model."
            raise ValueError(msg)
        module_name, class_name = _BUILTIN_MODELS[provider]
        chat_model = getattr(import_module(module_name), class_name)
    assert chat_model is not None
    signature = inspect.signature(chat_model)
    if not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    ):
        accepted = {
            p.name
            for p in signature.parameters.values()
            if p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        unsupported = kwargs.keys() - accepted
        if unsupported:
            # Report names only, never option values (which may contain credentials).
            msg = f"Provider {provider!r} does not accept arguments: {', '.join(sorted(unsupported))}."
            if unsupported_args == "raise":
                raise TypeError(msg)
            if unsupported_args == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)
            kwargs = {
                key: value for key, value in kwargs.items() if key not in unsupported
            }
    bound = signature.bind(model_name, **kwargs)
    return chat_model(*bound.args, **bound.kwargs)
