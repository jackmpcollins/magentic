# Vision

GPT-4 Vision can be used with magentic by using the `UserImageMessage` message type. This allows the LLM to accept images as input. Currently this is only supported with the OpenAI backend (`OpenaiChatModel` with `"gpt-4-vision-preview"`).

!!! note "Return types"

    GPT-4 Vision currently does not support function-calling/tools so functions using `@chatprompt` can only return `str`, `StreamedStr`, or `AsyncStreamedStr`.

!!! tip "`max_tokens`"

    By default `max_tokens` has a low value, so you will likely need to increase it.

For more information visit the [OpenAI Vision API documentation](https://platform.openai.com/docs/guides/vision).

## UserImageMessage

The `UserImageMessage` can be used with `@chatprompt` alongside other messages. The LLM must be set to OpenAI's GPT4 Vision model `OpenaiChatModel("gpt-4-vision-preview")`. This can be done by passing the `model` parameter to `@chatprompt`, or through the other methods of [configuration](configuration.md).

```Python
from magentic import chatprompt, OpenaiChatModel, UserMessage
from magentic.vision import UserImageMessage


IMAGE_URL_WOODEN_BOARDWALK = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


@chatprompt(
    UserMessage("Describe the following image in one sentence."),
    UserImageMessage(IMAGE_URL_WOODEN_BOARDWALK),
    model=OpenaiChatModel("gpt-4-vision-preview", max_tokens=2000),
)
def describe_image() -> str:
    ...


describe_image()
# 'A wooden boardwalk meanders through a lush green meadow under a blue sky with wispy clouds.'
```

## Placeholder

To provide the image as a function parameter, use `Placeholder`. This substitutes a function argument into the message when the function is called.

```Python hl_lines="10"
from magentic import chatprompt, OpenaiChatModel, Placeholder, UserMessage
from magentic.vision import UserImageMessage


IMAGE_URL_WOODEN_BOARDWALK = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


@chatprompt(
    UserMessage("Describe the following image in one sentence."),
    UserImageMessage(Placeholder(str, "image_url")),
    model=OpenaiChatModel("gpt-4-vision-preview", max_tokens=2000),
)
def describe_image(image_url: str) -> str:
    ...


describe_image(IMAGE_URL_WOODEN_BOARDWALK)
# 'A wooden boardwalk meanders through lush green wetlands under a partly cloudy blue sky.'
```

## bytes

`UserImageMessage` can also accept `bytes` as input. Like `str`, this can be passed directly or via `Placeholder`.

```python
import requests

from magentic import chatprompt, OpenaiChatModel, Placeholder, UserMessage
from magentic.vision import UserImageMessage


IMAGE_URL_WOODEN_BOARDWALK = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


def url_to_bytes(url: str) -> bytes:
    """Get the content of a URL as bytes."""
    return requests.get(url).content


@chatprompt(
    UserMessage("Describe the following image in one sentence."),
    UserImageMessage(Placeholder(bytes, "image_bytes")),
    model=OpenaiChatModel("gpt-4-vision-preview", max_tokens=2000),
)
def describe_image(image_bytes: bytes) -> str:
    ...


image_bytes = url_to_bytes(IMAGE_URL_WOODEN_BOARDWALK)
describe_image(image_bytes)
# 'The image shows a wooden boardwalk extending through a lush green wetland with a backdrop of blue skies and scattered clouds.'
```
