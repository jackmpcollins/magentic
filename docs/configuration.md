# LLM Configuration

Currently two backends are available

- `openai` : the default backend that uses the `openai` Python package. Supports all features.
- `litellm` : uses the `litellm` Python package to enable querying LLMs from [many different providers](https://docs.litellm.ai/docs/providers). Install this with `pip install magentic[litellm]`. Note: some models may not support all features of `magentic` e.g. function calling/structured output and streaming.

The backend and LLM used by `magentic` can be configured in several ways. The order of precedence of configuration is

1. Arguments explicitly passed when initializing an instance in Python
1. Values set using a context manager in Python
1. Environment variables
1. Default values from [src/magentic/settings.py](https://github.com/jackmpcollins/magentic/src/magentic/settings.py)

```python
from magentic import OpenaiChatModel, prompt
from magentic.chat_model.litellm_chat_model import LitellmChatModel


@prompt("Say hello")
def say_hello() -> str: ...


@prompt(
    "Say hello",
    model=LitellmChatModel("ollama/llama2"),
)
def say_hello_litellm() -> str: ...


say_hello()  # Uses env vars or default settings

with OpenaiChatModel("gpt-3.5-turbo", temperature=1):
    say_hello()  # Uses openai with gpt-3.5-turbo and temperature=1 due to context manager
    say_hello_litellm()  # Uses litellm with ollama/llama2 because explicitly configured
```

The following environment variables can be set.

| Environment Variable         | Description                            | Example                |
| ---------------------------- | -------------------------------------- | ---------------------- |
| MAGENTIC_BACKEND             | The package to use as the LLM backend  | openai                 |
| MAGENTIC_LITELLM_MODEL       | LiteLLM model                          | claude-2               |
| MAGENTIC_LITELLM_API_BASE    | The base url to query                  | http://localhost:11434 |
| MAGENTIC_LITELLM_MAX_TOKENS  | LiteLLM max number of generated tokens | 1024                   |
| MAGENTIC_LITELLM_TEMPERATURE | LiteLLM temperature                    | 0.5                    |
| MAGENTIC_OPENAI_MODEL        | OpenAI model                           | gpt-4                  |
| MAGENTIC_OPENAI_API_KEY      | OpenAI API key to be used by magentic  | sk-...                 |
| MAGENTIC_OPENAI_API_TYPE     | Allowed options: "openai", "azure"     | azure                  |
| MAGENTIC_OPENAI_BASE_URL     | Base URL for an OpenAI-compatible API  | http://localhost:8080  |
| MAGENTIC_OPENAI_MAX_TOKENS   | OpenAI max number of generated tokens  | 1024                   |
| MAGENTIC_OPENAI_SEED         | Seed for deterministic sampling        | 42                     |
| MAGENTIC_OPENAI_TEMPERATURE  | OpenAI temperature                     | 0.5                    |

When using the `openai` backend, setting the `MAGENTIC_OPENAI_BASE_URL` environment variable or using `OpenaiChatModel(..., base_url="http://localhost:8080")` in code allows you to use `magentic` with any OpenAI-compatible API e.g. [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line&pivots=programming-language-python#create-a-new-python-application), [LiteLLM OpenAI Proxy Server](https://docs.litellm.ai/docs/proxy_server), [LocalAI](https://localai.io/howtos/easy-request-openai/). Note that if the API does not support function calling then you will not be able to create prompt-functions that return Python objects, but other features of `magentic` will still work.

To use Azure with the openai backend you will need to set the `MAGENTIC_OPENAI_API_TYPE` environment variable to "azure" or use `OpenaiChatModel(..., api_type="azure")`, and also set the environment variables needed by the openai package to access Azure. See https://github.com/openai/openai-python#microsoft-azure-openai
