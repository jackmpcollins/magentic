from typing import Any, Generic, TypeVar, overload

from magentic.chat_model.message import Message, Placeholder

T = TypeVar("T", bytes, str)
ImageContentT = TypeVar("ImageContentT")


class UserImageMessage(Message[ImageContentT], Generic[ImageContentT]):
    """A message containing an image sent by a user to an LLM chat model."""

    def __init__(self, content: ImageContentT, **data: Any):
        super().__init__(content=content, **data)

    @overload
    def format(
        self: "UserImageMessage[Placeholder[T]]", **kwargs: Any
    ) -> "UserImageMessage[T]": ...

    @overload
    def format(self: "UserImageMessage[T]", **kwargs: Any) -> "UserImageMessage[T]": ...

    def format(
        self: "UserImageMessage[Placeholder[T]] | UserImageMessage[T]",
        **kwargs: Any,
    ) -> "UserImageMessage[T]":
        """Format the message using the provided substitutions."""
        if isinstance(self.content, Placeholder):
            return UserImageMessage(self.content.format(**kwargs))
        return UserImageMessage(self.content)
