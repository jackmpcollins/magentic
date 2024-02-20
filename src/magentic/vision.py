import base64
from typing import Any, Generic, Literal, Sequence, TypeVar, cast, overload

from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from pydantic import BaseModel

from magentic.chat_model.message import Message
from magentic.chat_model.openai_chat_model import (
    OpenaiMessageRole,
    message_to_openai_message,
)


class ImageUrl(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


PartContentT = TypeVar("PartContentT", bound=str | ImageUrl)


class MultipartUserMessage(Message[Sequence[PartContentT]], Generic[PartContentT]):
    @staticmethod
    def _format_content_item(item: PartContentT, **kwargs: Any) -> PartContentT:
        if isinstance(item, str):
            # Cast back to ContentT to satisfy mypy
            return cast(PartContentT, item.format(**kwargs))
        return item

    def format(self, **kwargs: Any) -> "MultipartUserMessage[PartContentT]":
        content = [self._format_content_item(item, **kwargs) for item in self.content]
        return MultipartUserMessage(content)


@message_to_openai_message.register(MultipartUserMessage)
def _(message: MultipartUserMessage[Any]) -> ChatCompletionMessageParam:
    content: list[ChatCompletionContentPartParam] = []
    for content_item in message.content:
        if isinstance(content_item, str):
            content.append({"type": "text", "text": content_item})
        elif isinstance(content_item, ImageUrl):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": content_item.url,
                        "detail": content_item.detail,
                    },
                }
            )
        else:
            msg = f"Unsupported content type: {type(content_item)}"
            raise TypeError(msg)

    return {"role": OpenaiMessageRole.USER.value, "content": content}


ImageContentT = TypeVar("ImageContentT", bound=bytes | str)


class ImageUserMessage(Message[ImageContentT]):
    @overload
    def format(
        self: "ImageUserMessage[bytes]", **kwargs: Any
    ) -> "ImageUserMessage[bytes]":
        ...

    @overload
    def format(
        self: "ImageUserMessage[str]", **kwargs: Any
    ) -> "ImageUserMessage[bytes] | ImageUserMessage[str]":
        ...

    def format(
        self: "ImageUserMessage[bytes] | ImageUserMessage[str]", **kwargs: Any
    ) -> "ImageUserMessage[bytes] | ImageUserMessage[str]":
        """Format the message using the given function arguments.

        - If existing content is `bytes`, it is returned as is.
        - If existing content is a string
            - Check if the string is a key in `kwargs` and use the value from `kwargs` if it is.
            - Call `str.format` with the given `kwargs` and return the result.
        """
        if isinstance(self.content, bytes):
            return ImageUserMessage(self.content)
        if isinstance(self.content, str):
            if self.content in kwargs:
                if not isinstance(kwargs[self.content], bytes | str):
                    msg = f"Unsupported content type: {type(kwargs[self.content])}"
                    raise TypeError(msg)
                return ImageUserMessage(kwargs[self.content])
            return ImageUserMessage(self.content.format(**kwargs))
        msg = f"Unsupported content type: {type(self.content)}"
        raise TypeError(msg)


@message_to_openai_message.register(ImageUserMessage)
def _(message: ImageUserMessage[bytes | str]) -> ChatCompletionMessageParam:
    if isinstance(message.content, str):
        url = message.content
    else:
        base64_image = base64.b64encode(message.content).decode("utf-8")
        url = f"data:image/jpeg;base64,{base64_image}"

    return {
        "role": OpenaiMessageRole.USER.value,
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": url,
                    "detail": "auto",
                },
            }
        ],
    }
