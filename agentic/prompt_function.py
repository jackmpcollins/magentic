from functools import wraps
from inspect import Parameter, Signature
from typing import Generic, Sequence, Type, TypeVar

import openai
from pydantic.generics import GenericModel


T = TypeVar("T")


class Output(GenericModel, Generic[T]):
    output: T


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
        output_model = Output[self._signature.return_annotation]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {
                    "role": "user",
                    "content": self._template.format(**bound_args.arguments),
                },
            ],
            functions=[
                {
                    "name": "output",
                    "description": "Return the output",
                    "parameters": output_model.schema(),
                }
            ],
            function_call={"name": "output"},
            temperature=0,
        )
        output = output_model.parse_raw(
            response["choices"][0]["message"]["function_call"]["arguments"]
        )
        return output.output


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
