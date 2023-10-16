from magentic.chat_model.base import ChatModel
from magentic.settings import Backend, get_settings


def get_chat_model() -> ChatModel:
    settings = get_settings()

    match settings.backend:
        case Backend.OPENAI:
            from magentic.chat_model.openai_chat_model import OpenaiChatModel

            return OpenaiChatModel(
                model=settings.openai_model,
                max_tokens=settings.openai_max_tokens,
                temperature=settings.openai_temperature,
            )
        case _:
            msg = f"Backend {settings.backend} does not support chat model."
            raise NotImplementedError(msg)
