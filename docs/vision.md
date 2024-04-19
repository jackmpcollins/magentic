# Vision

GPT-4 Vision can be used with magentic by using the `UserImageMessage` message type. This allows the LLM to accept images as input. Currently this is only supported with the OpenAI backend (`OpenaiChatModel`).

!!! note "Return types"

    `gpt-4-vision-preview` does not support function-calling/tools so only `str`, `StreamedStr`, and `AsyncStreamedStr` work as return types.

!!! tip "`max_tokens`"

    By default, `gpt-4-vision-preview` has a low value for `max_tokens` so you will likely need to increase it.

For more information visit the [OpenAI Vision API documentation](https://platform.openai.com/docs/guides/vision).

## UserImageMessage

The `UserImageMessage` can be used in `@chatprompt` alongside other messages. The LLM must be set to an OpenAI model that supports vision, currently `gpt-4-vision-preview` and `gpt-4-turbo` (the default `ChatModel`). This can be done by passing the `model` parameter to `@chatprompt`, or through the other methods of [configuration](configuration.md).

```python
from pydantic import BaseModel, Field

from magentic import chatprompt, UserMessage
from magentic.vision import UserImageMessage


IMAGE_URL_WOODEN_BOARDWALK = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


class ImageDetails(BaseModel):
    description: str = Field(description="A brief description of the image.")
    name: str = Field(description="A short name.")


@chatprompt(
    UserMessage("Describe the following image in one sentence."),
    UserImageMessage(IMAGE_URL_WOODEN_BOARDWALK),
)
def describe_image() -> ImageDetails: ...


image_details = describe_image()
print(image_details.name)
# 'Wooden Boardwalk in Green Wetland'
print(image_details.description)
# 'A serene wooden boardwalk meanders through a lush green wetland under a blue sky dotted with clouds.'
```

For more info on the `@chatprompt` decorator, see [Chat Prompting](chat-prompting.md).

## Placeholder

In the previous example, the image url was tied to the function. To provide the image as a function parameter, use `Placeholder`. This substitutes a function argument into the message when the function is called.

```python hl_lines="10"
from magentic import chatprompt, Placeholder, UserMessage
from magentic.vision import UserImageMessage


IMAGE_URL_WOODEN_BOARDWALK = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


@chatprompt(
    UserMessage("Describe the following image in one sentence."),
    UserImageMessage(Placeholder(str, "image_url")),
)
def describe_image(image_url: str) -> str: ...


describe_image(IMAGE_URL_WOODEN_BOARDWALK)
# 'A wooden boardwalk meanders through lush green wetlands under a partly cloudy blue sky.'
```

## bytes

`UserImageMessage` can also accept `bytes` as input. Like `str`, this can be passed directly or via `Placeholder`.

```python
import requests

from magentic import chatprompt, Placeholder, UserMessage
from magentic.vision import UserImageMessage


IMAGE_URL_WOODEN_BOARDWALK = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


def url_to_bytes(url: str) -> bytes:
    """Get the content of a URL as bytes."""

    # A custom user-agent is necessary to comply with Wikimedia user-agent policy
    # https://meta.wikimedia.org/wiki/User-Agent_policy
    headers = {"User-Agent": "MagenticExampleBot (https://magentic.dev/)"}
    return requests.get(url, headers=headers, timeout=10).content


@chatprompt(
    UserMessage("Describe the following image in one sentence."),
    UserImageMessage(Placeholder(bytes, "image_bytes")),
)
def describe_image(image_bytes: bytes) -> str: ...


image_bytes = url_to_bytes(IMAGE_URL_WOODEN_BOARDWALK)
describe_image(image_bytes)
# 'The image shows a wooden boardwalk extending through a lush green wetland with a backdrop of blue skies and scattered clouds.'
```
