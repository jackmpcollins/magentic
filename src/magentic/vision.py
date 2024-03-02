import base64
from typing import Any, Generic, TypeVar, overload

from openai.types.chat import ChatCompletionMessageParam

from magentic.chat_model.message import Message, Placeholder
from magentic.chat_model.openai_chat_model import (
    OpenaiMessageRole,
    message_to_openai_message,
)

T = TypeVar("T")
ImageContentT = TypeVar("ImageContentT")


class ImageUserMessage(Message[ImageContentT], Generic[ImageContentT]):
    @overload
    def format(
        self: "ImageUserMessage[Placeholder[T]]", **kwargs: Any
    ) -> "ImageUserMessage[T]":
        ...

    @overload
    def format(self: "ImageUserMessage[T]", **kwargs: Any) -> "ImageUserMessage[T]":
        ...

    def format(
        self: "ImageUserMessage[Placeholder[T]] | ImageUserMessage[T]",
        **kwargs: Any,
    ) -> "ImageUserMessage[T]":
        """Format the message using the given function arguments."""
        if isinstance(self.content, Placeholder):
            return ImageUserMessage(self.content.format(**kwargs))
        return ImageUserMessage(self.content)


@message_to_openai_message.register(ImageUserMessage)
def _(
    message: ImageUserMessage[bytes] | ImageUserMessage[str],
) -> ChatCompletionMessageParam:
    if isinstance(message.content, bytes):
        base64_image = base64.b64encode(message.content).decode("utf-8")
        url = f"data:image/jpeg;base64,{base64_image}"
    elif isinstance(message.content, str):
        url = message.content
    else:
        msg = f"Invalid content type: {type(message.content)}"
        raise TypeError(msg)

    return {
        "role": OpenaiMessageRole.USER.value,
        "content": [{"type": "image_url", "image_url": {"url": url, "detail": "auto"}}],
    }
