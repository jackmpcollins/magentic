from typing import Any, Generic, Literal, Sequence, TypeVar, cast

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


ContentT = TypeVar("ContentT")


class MultipartUserMessage(Message[Sequence[ContentT]], Generic[ContentT]):
    @staticmethod
    def _format_content_item(item: ContentT, **kwargs: Any) -> ContentT:
        if isinstance(item, str):
            # Cast back to ContentT to satisfy mypy
            return cast(ContentT, item.format(**kwargs))
        return item

    def format(self, **kwargs: Any) -> "MultipartUserMessage[ContentT]":
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
