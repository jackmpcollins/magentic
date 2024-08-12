# Retrying

## LLM-Assisted Retries

Occasionally the LLM returns an output that cannot be parsed into any of the output types or function calls that were requested. Additionally, the pydantic models you define might have extra validation that is not represented by the type annotations alone. In these cases, LLM-assisted retries can be used to automatically resubmit the output as well as the associated error message back to the LLM, giving it another opportunity with more information to meet the output schema requirements.

To enable retries, simply set the `max_retries` parameter to a non-zero value in `@prompt` or `@chatprompt`.

In this example

- the LLM first returns a country that is not Ireland
- then the pydantic model validation fails with error "Country must be Ireland"
- the original output as well as a message containing the error are resubmitted to the LLM
- the LLM correctly meets the output requirement returning "Ireland"

```python
from typing import Annotated

from magentic import prompt
from pydantic import AfterValidator, BaseModel


def assert_is_ireland(v: str) -> str:
    if v != "Ireland":
        raise ValueError("Country must be Ireland")
    return v


class Country(BaseModel):
    name: Annotated[str, AfterValidator(assert_is_ireland)]
    capital: str


@prompt(
    "Return a country",
    max_retries=3,
)
def get_country() -> Country: ...


get_country()
# 05:13:55.607 Calling prompt-function get_country
# 05:13:55.622   LLM-assisted retries enabled. Max 3
# 05:13:55.627     Chat Completion with 'gpt-4o' [LLM]
# 05:13:56.309     streaming response from 'gpt-4o' took 0.11s [LLM]
# 05:13:56.310     Retrying Chat Completion. Attempt 1.
# 05:13:56.322     Chat Completion with 'gpt-4o' [LLM]
# 05:13:57.456     streaming response from 'gpt-4o' took 0.00s [LLM]
#
# Country(name='Ireland', capital='Dublin')
```

LLM-Assisted retries are intended to address cases where the LLM failed to generate valid output. Errors due to LLM provider rate limiting, internet connectivity issues, or other issues that cannot be solved by reprompting the LLM should be handled using other methods. For example [jd/tenacity](https://github.com/jd/tenacity) or [hynek/stamina](https://github.com/hynek/stamina) to retry a Python function.

### RetryChatModel

Under the hood, LLM-assisted retries are implemented using the `RetryChatModel` which wraps any other `ChatModel`, catches exceptions, and resubmits them to the LLM. To implement your own retry handling you can follow the pattern of this class. Please file a [GitHub issue](https://github.com/jackmpcollins/magentic/issues) if you encounter exceptions that should be included in the LLM-assisted retries.

To use the `RetryChatModel` directly rather than via the `max_retries` parameter, simply pass it as the `model` argument to the decorator. Extending the example above

```python
from magentic import OpenaiChatModel
from magentic.chat_model.retry_chat_model import RetryChatModel


@prompt(
    "Return a country",
    model=RetryChatModel(OpenaiChatModel("gpt-4o-mini"), max_retries=3),
)
def get_country() -> Country: ...


get_country()
```
