import importlib
from typing import Any
from unittest.mock import Mock

import pytest

from magentic import Chat, chatprompt, model, prompt, prompt_chain, register_chat_model
from magentic.backend import get_chat_model
from magentic.chat_model.base import ChatModel
from magentic.chat_model.message import AssistantMessage, UserMessage
from magentic.chat_model.openai_chat_model import OpenaiChatModel

factory_module = importlib.import_module("magentic.model")


class CustomModel(ChatModel):
    def __init__(self, name: str, *, temperature: float = 0.5):
        self.name = name
        self.temperature = temperature

    def complete(self, messages, functions=None, output_types=None, *, stop=None):
        return AssistantMessage(self.name)

    async def acomplete(
        self, messages, functions=None, output_types=None, *, stop=None
    ):
        return AssistantMessage(self.name)


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    monkeypatch.setattr(factory_module, "_REGISTERED_MODELS", {})


@pytest.mark.parametrize(
    "provider", ["openai", "anthropic", "litellm", "mistral", "openrouter"]
)
def test_builtin_factory_lazy_constructor(provider, monkeypatch):
    constructor = Mock(spec=lambda model, *, temperature=None: None)

    # Give inspect an actual class constructor while capturing arguments, without SDKs.
    class Captured(CustomModel):
        def __init__(self, name: str, *, temperature: float = 0.5):
            constructor(name, temperature=temperature)
            super().__init__(name, temperature=temperature)

    import_ = Mock(
        return_value=type(
            "Module", (), {factory_module._BUILTIN_MODELS[provider][1]: Captured}
        )
    )
    monkeypatch.setattr(factory_module, "import_module", import_)
    instance = model(f"{provider}:family:model", temperature=0.2)
    assert isinstance(instance, Captured)
    constructor.assert_called_once_with("family:model", temperature=0.2)
    import_.assert_called_once_with(factory_module._BUILTIN_MODELS[provider][0])


@pytest.mark.parametrize(
    "name", ["", "openai", ":model", "openai:", " openai:model", "unknown:model"]
)
def test_invalid_names_before_import(name, monkeypatch):
    import_ = Mock()
    monkeypatch.setattr(factory_module, "import_module", import_)
    with pytest.raises(ValueError, match="form|provider|registered|Provider"):
        model(name)
    import_.assert_not_called()


@pytest.mark.parametrize("name", ["", "two:parts", "white space", "openai"])
def test_invalid_or_builtin_registration(name):
    with pytest.raises(ValueError, match="form|provider|registered|Provider"):
        register_chat_model(name, CustomModel)


def test_custom_registration_and_duplicate():
    register_chat_model("custom", CustomModel)
    instance = model("custom:local", temperature=0.9)
    assert isinstance(instance, CustomModel)
    assert (instance.name, instance.temperature) == ("local", 0.9)
    with pytest.raises(ValueError, match="already registered"):
        register_chat_model("custom", CustomModel)
    with pytest.raises(TypeError, match="ChatModel subclass"):
        register_chat_model("not-model", str)  # type: ignore[arg-type]


@pytest.mark.parametrize("policy", ["raise", "warn", "ignore"])
def test_unsupported_keyword_policy(policy):
    register_chat_model("custom", CustomModel)
    if policy == "raise":
        with pytest.raises(TypeError, match="unknown") as error:
            model("custom:local", unknown="secret-value")
        assert "secret-value" not in str(error.value)
    elif policy == "warn":
        with pytest.warns(UserWarning, match="unknown") as warning:
            instance = model(
                "custom:local", unsupported_args="warn", unknown="secret-value"
            )
        assert isinstance(instance, CustomModel)
        assert "secret-value" not in str(warning[0].message)
    else:
        assert isinstance(
            model("custom:local", unsupported_args="ignore", unknown=True), CustomModel
        )


def test_no_constructor_typeerror_swallowed_or_retried():
    calls = []

    class Broken(CustomModel):
        def __init__(self, name: str, **kwargs: Any):
            calls.append(kwargs)
            msg = "constructor failed internally"
            raise TypeError(msg)

    register_chat_model("broken", Broken)
    with pytest.raises(TypeError, match="failed internally"):
        model("broken:local", unsupported_args="ignore", arbitrary=True)
    assert calls == [{"arbitrary": True}]


def test_kwargs_custom_constructor_owns_validation():
    class Flexible(CustomModel):
        def __init__(self, name: str, **kwargs: Any):
            super().__init__(name)
            self.options = kwargs

    register_chat_model("flexible", Flexible)
    instance = model("flexible:local", extra=1)
    assert isinstance(instance, Flexible)
    assert instance.options == {"extra": 1}


def test_dependency_error_preserved(monkeypatch):
    import_ = Mock(side_effect=ImportError("Install magentic[anthropic]"))
    monkeypatch.setattr(factory_module, "import_module", import_)
    with pytest.raises(ImportError, match=r"magentic\[anthropic\]"):
        model("anthropic:local")


def test_environment_uses_factory(monkeypatch):
    create = Mock(return_value=CustomModel("env"))
    monkeypatch.setenv("MAGENTIC_BACKEND", "openai")
    monkeypatch.setenv("MAGENTIC_OPENAI_MODEL", "environment-model")
    monkeypatch.setenv("MAGENTIC_OPENAI_TEMPERATURE", "0.7")
    monkeypatch.setattr("magentic.backend.model", create)
    instance = get_chat_model()
    assert isinstance(instance, CustomModel)
    assert instance.name == "env"
    assert create.call_args.args == ("openai:environment-model",)
    assert create.call_args.kwargs["temperature"] == 0.7


@pytest.mark.asyncio
async def test_string_models_all_surfaces_and_context_precedence():
    register_chat_model("custom", CustomModel)

    @prompt("hello", model="custom:prompt")
    def sync() -> str: ...
    @prompt("hello", model="custom:async")
    async def async_() -> str: ...
    @chatprompt(UserMessage("hello"), model="custom:chatprompt")
    def chat_sync() -> str: ...
    @chatprompt(UserMessage("hello"), model="custom:asyncchat")
    async def chat_async() -> str: ...
    @prompt_chain("hello", model="custom:chain")
    def chain() -> str: ...
    @prompt_chain("hello", model="custom:asyncchain")
    async def async_chain() -> str: ...
    @prompt("hello")
    def default() -> str: ...

    with CustomModel("context"):
        assert default() == "context"
        assert sync() == "prompt"
        assert await async_() == "async"
        assert chat_sync() == "chatprompt"
        assert await chat_async() == "asyncchat"
        assert chain() == "chain"
        assert await async_chain() == "asyncchain"
        assert Chat(model="custom:chat").submit().last_message.content == "chat"
        assert (
            await Chat(model="custom:asyncchat").asubmit()
        ).last_message.content == "asyncchat"
        current = get_chat_model()
        assert isinstance(current, CustomModel)
        assert current.name == "context"


def test_factory_real_openai_constructor_without_request():
    instance = model("openai:gpt-test", api_key="test", temperature=0.1)
    assert isinstance(instance, OpenaiChatModel)
    assert instance.model == "gpt-test"
    assert instance.temperature == 0.1


def test_invalid_policy_before_import(monkeypatch):
    import_ = Mock()
    monkeypatch.setattr(factory_module, "import_module", import_)
    with pytest.raises(ValueError, match="unsupported_args"):
        model("openai:local", unsupported_args="drop")  # type: ignore[arg-type]
    import_.assert_not_called()


def test_missing_required_and_duplicate_model_argument_not_hidden():
    class Required(CustomModel):
        def __init__(self, name: str, *, required: int):
            super().__init__(name)

    register_chat_model("required", Required)
    with pytest.raises(TypeError, match="required"):
        model("required:local", unsupported_args="ignore")
    with pytest.raises(TypeError, match="multiple values"):
        model(
            "openai:local", model="duplicate", api_key="test", unsupported_args="ignore"
        )


def test_default_policy_validates_before_constructor():
    calls = []

    class Captured(CustomModel):
        def __init__(self, name: str):
            calls.append(name)

    register_chat_model("captured", Captured)
    with pytest.raises(TypeError, match="unknown"):
        model("captured:local", unknown=True)
    assert calls == []


@pytest.mark.parametrize(
    ("dependency", "provider", "extra"),
    [("anthropic", "anthropic", "anthropic"), ("litellm", "litellm", "litellm")],
)
def test_actual_optional_dependency_import_error(dependency, provider, extra):
    import subprocess
    import sys

    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == sys.argv[1] or fullname.startswith(sys.argv[1] + '.'):
            raise ModuleNotFoundError('blocked optional dependency', name=sys.argv[1])
sys.meta_path.insert(0, Block())
from magentic import model
try:
    model(sys.argv[2] + ':local')
except ImportError as error:
    print(str(error))
else:
    raise AssertionError('missing dependency did not fail')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, dependency, provider],
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"magentic[{extra}]" in result.stdout


@pytest.mark.asyncio
async def test_string_model_retry_and_object_unchanged():
    register_chat_model("custom", CustomModel)

    @prompt("hello", model="custom:retry", max_retries=1)
    def sync() -> str: ...
    @chatprompt(UserMessage("hello"), model="custom:retry-async", max_retries=1)
    async def async_() -> str: ...

    assert sync() == "retry"
    assert await async_() == "retry-async"
    original = CustomModel("object")
    assert Chat(model=original).model is original

    @prompt("hello", model=original)
    def explicit() -> str: ...

    assert explicit.model is original
    with model("custom:outer") as result:
        # Preserve the existing ChatModel.__enter__ result (None).
        assert result is None
        with model("custom:inner"):
            current = get_chat_model()
            assert isinstance(current, CustomModel)
            assert current.name == "inner"
        current = get_chat_model()
        assert isinstance(current, CustomModel)
        assert current.name == "outer"


def test_custom_positional_only_and_keyword_options():
    class Positional(CustomModel):
        def __init__(self, name: str, /, *, temperature: float = 0.5):
            super().__init__(name, temperature=temperature)

    register_chat_model("positional", Positional)
    instance = model("positional:local", temperature=0.7)
    assert isinstance(instance, Positional)
    assert instance.temperature == 0.7
