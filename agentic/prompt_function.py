from functools import wraps
from inspect import Parameter, Signature
from typing import Sequence, Type

import openai


class PromptFunction:
    def __init__(
        self,
        parameters: Sequence[Parameter],
        return_type: Type,
        template: str,
    ):
        self._signature = Signature(
            parameters=parameters,
            return_annotation=return_type,
        )
        self._template = template

    def __call__(self, *args, **kwargs):
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": self._template.format(**bound_args.arguments),
                },
            ],
            temperature=0,
        )
        output: str = response["choices"][0]["message"]["content"]
        return output


def prompt():
    def decorator(func):
        func_signature = Signature.from_callable(func)
        return wraps(func)(
            PromptFunction(
                parameters=list(func_signature.parameters.values()),
                return_type=func_signature.return_annotation,
                template=func.__doc__,
            )
        )

    return decorator
