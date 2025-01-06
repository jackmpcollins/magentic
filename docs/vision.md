# Vision

Image inputs can be provided to LLMs in magentic by using `ImageBytes` or `ImageUrl` within the `UserMessage` message type. The LLM used must support vision, for example `gpt-4o` (the default `ChatModel`). The model can be set by passing the `model` parameter to `@chatprompt`, or through the other methods of [configuration](configuration.md).

!!! note "Anthropic Image URLs"

    Anthropic models currently do not support supplying an image as a url, just bytes.

For more information visit the [OpenAI Vision API documentation](https://platform.openai.com/docs/guides/vision) or the [Anthropic Vision API documentation](https://docs.anthropic.com/en/docs/build-with-claude/vision#example-multiple-images).

!!! warning "UserImageMessage Deprecation"

    Previously the `UserImageMessage` was used for vision capabilities. This is now deprecated and will be removed in a future version of Magentic. It is recommended to use `ImageBytes` or `ImageUrl` within the `UserMessage` message type instead to ensure compatibility with future updates.

## ImageUrl

As shown in [Chat Prompting](chat-prompting.md), `@chatprompt` can be used to supply a group of messages as a prompt to the LLM. `UserMessage` accepts a sequence of content blocks as input, which can be `str`, `ImageBytes`, `ImageUrl`, or other content types. `ImageUrl` is used to provide an image url to the LLM.

```python
from pydantic import BaseModel, Field

from magentic import chatprompt, ImageUrl, UserMessage


IMAGE_URL_WOODEN_BOARDWALK = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


class ImageDetails(BaseModel):
    description: str = Field(description="A brief description of the image.")
    name: str = Field(description="A short name.")


@chatprompt(
    UserMessage(
        [
            "Describe the following image in one sentence.",
            ImageUrl(IMAGE_URL_WOODEN_BOARDWALK),
        ]
    ),
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

In the previous example, the image url was tied to the function. To provide the image as a function parameter, use `Placeholder`. This substitutes a function argument into the message when the function is called. The placeholder will also automatically coerce the argument to the correct type if possible, for example `str` to `ImageUrl`.

```python hl_lines="10"
from magentic import chatprompt, ImageUrl, Placeholder, UserMessage


IMAGE_URL_WOODEN_BOARDWALK = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


@chatprompt(
    UserMessage(
        [
            "Describe the following image in one sentence.",
            Placeholder(ImageUrl, "image_url"),
        ]
    ),
)
def describe_image(image_url: str) -> str: ...


describe_image(IMAGE_URL_WOODEN_BOARDWALK)
# 'A wooden boardwalk meanders through lush green wetlands under a partly cloudy blue sky.'
```

## ImageBytes

`UserMessage` can also accept `ImageBytes` as a content block. Like `ImageUrl`, this can be passed directly or via `Placeholder`.

```python
import requests

from magentic import chatprompt, ImageBytes, Placeholder, UserMessage


IMAGE_URL_WOODEN_BOARDWALK = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


def url_to_bytes(url: str) -> bytes:
    """Get the content of a URL as bytes."""

    # A custom user-agent is necessary to comply with Wikimedia user-agent policy
    # https://meta.wikimedia.org/wiki/User-Agent_policy
    headers = {"User-Agent": "MagenticExampleBot (https://magentic.dev/)"}
    return requests.get(url, headers=headers, timeout=10).content


@chatprompt(
    UserMessage(
        [
            "Describe the following image in one sentence.",
            Placeholder(ImageBytes, "image_bytes"),
        ]
    ),
)
def describe_image(image_bytes: bytes) -> str: ...


image_bytes = url_to_bytes(IMAGE_URL_WOODEN_BOARDWALK)
describe_image(image_bytes)
# 'The image shows a wooden boardwalk extending through a lush green wetland with a backdrop of blue skies and scattered clouds.'
```
