from magentic.chat_model.base import ChatModel, _chat_model_context
from magentic.settings import Backend, get_settings


def get_chat_model() -> ChatModel:
    if chat_model := _chat_model_context.get():
        return chat_model

    settings = get_settings()

    match settings.backend:
        case Backend.ANTHROPIC:
            from magentic.chat_model.anthropic_chat_model import AnthropicChatModel

            return AnthropicChatModel(
                model=settings.anthropic_model,
                api_key=settings.anthropic_api_key,
                base_url=settings.anthropic_base_url,
                max_tokens=settings.anthropic_max_tokens,
                temperature=settings.anthropic_temperature,
            )
        case Backend.LITELLM:
            from magentic.chat_model.litellm_chat_model import LitellmChatModel

            return LitellmChatModel(
                model=settings.litellm_model,
                api_base=settings.litellm_api_base,
                max_tokens=settings.litellm_max_tokens,
                temperature=settings.litellm_temperature,
            )
        case Backend.MISTRAL:
            from magentic.chat_model.mistral_chat_model import MistralChatModel

            return MistralChatModel(
                model=settings.mistral_model,
                api_key=settings.mistral_api_key,
                base_url=settings.mistral_base_url,
                max_tokens=settings.mistral_max_tokens,
                seed=settings.mistral_seed,
                temperature=settings.mistral_temperature,
            )
        case Backend.OPENAI:
            from magentic.chat_model.openai_chat_model import OpenaiChatModel

            return OpenaiChatModel(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                api_type=settings.openai_api_type,
                base_url=settings.openai_base_url,
                max_tokens=settings.openai_max_tokens,
                seed=settings.openai_seed,
                temperature=settings.openai_temperature,
            )
        case _:
            msg = f"Backend {settings.backend} does not support chat model."
            raise NotImplementedError(msg)
