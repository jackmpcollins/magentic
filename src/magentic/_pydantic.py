from pydantic import ConfigDict as _ConfigDict


class ConfigDict(_ConfigDict, total=False):
    openai_strict: bool
