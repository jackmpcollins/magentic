# Chat

This page covers the `Chat` class which can be used to manage an ongoing conversation with an LLM chat model. To define a reusable LLM query template use the `@chatprompt` decorator instead, see [Chat Prompting](chat-prompting.md).

The `Chat` class represents an ongoing conversation with an LLM. It keeps track of the messages exchanged between the user and the model and allows submitting the conversation to the model to get a response.

## Basic Usage

```python
from magentic import Chat, OpenaiChatModel, UserMessage

# Create a new Chat instance
chat = Chat(
    messages=[UserMessage("Say hello")],
    model=OpenaiChatModel("gpt-4o"),
)

# Append a new user message
chat = chat.add_user_message("Actually, say goodbye!")
print(chat.messages)
# [UserMessage('Say hello'), UserMessage('Actually, say goodbye!')]

# Submit the chat to the LLM to get a response
chat = chat.submit()
print(chat.last_message.content)
# 'Hello! Just kidding—goodbye!'
```

Note that all methods of `Chat` return a new `Chat` instance with the updated messages. This allows branching the conversation and keeping track of multiple conversation paths.

The following methods are available to manually add messages to the chat by providing just the content of the message:

- `add_system_message`: Adds a system message to the chat.
- `add_user_message`: Adds a user message to the chat.
- `add_assistant_message`: Adds an assistant message to the chat.

There is also a generic `add_message` method to add a `Message` object to the chat. And the `submit` method is used to submit the chat to the LLM model which adds an `AssistantMessage` to the chat containing the model's response.

## Function Calling

Function calling can be done with the `Chat` class by providing the list of functions when creating the instance, similar to the `@chatprompt` decorator. Similarly, structured outputs can be returned by setting the `output_types` parameter.

If the last message in the chat is an `AssistantMessage` containing a `FunctionCall` or `ParallelFunctionCall`, calling the `exec_function_call` method will execute the function call(s) and append the result(s) to the chat. Then, if needed, the chat can be submitted to the LLM again to get another response.

```python hl_lines="23-25"
from magentic import (
    AssistantMessage,
    Chat,
    FunctionCall,
    OpenaiChatModel,
    UserMessage,
)


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    # Pretend to query an API
    return {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }


chat = Chat(
    messages=[UserMessage("What's the weather like in Boston?")],
    functions=[get_current_weather],
    # `FunctionCall` must be in output_types to get `FunctionCall` outputs
    output_types=[FunctionCall, str],
    model=OpenaiChatModel("gpt-4o"),
)
chat = chat.submit()
print(chat.messages)
# [UserMessage("What's the weather like in Boston?"),
#  AssistantMessage(FunctionCall(<function get_current_weather at 0x130a92160>, 'Boston'))]

# Execute the function call and append the result to the chat
chat = chat.exec_function_call()
print(chat.messages)
# [UserMessage("What's the weather like in Boston?"),
#  AssistantMessage(FunctionCall(<function get_current_weather at 0x130a92160>, 'Boston')),
#  FunctionResultMessage({'location': 'Boston', 'temperature': '72', 'unit': 'fahrenheit', 'forecast': ['sunny', 'windy']},
#                        FunctionCall(<function get_current_weather at 0x130a92160>, 'Boston'))]

# Submit the chat again to get the final LLM response
chat = chat.submit()
print(chat.messages)
# [UserMessage("What's the weather like in Boston?"),
#  AssistantMessage(FunctionCall(<function get_current_weather at 0x130a92160>, 'Boston')),
#  FunctionResultMessage({'location': 'Boston', 'temperature': '72', 'unit': 'fahrenheit', 'forecast': ['sunny', 'windy']},
#                        FunctionCall(<function get_current_weather at 0x130a92160>, 'Boston')),
#  AssistantMessage("The current weather in Boston is 72°F, and it's sunny with windy conditions.")]
```

## Streaming

Streaming types such as `StreamedStr`, `StreamedOutput`, and `Iterable[T]` can be provided in the `output_types` parameter. When the `.submit()` method is called, an `AssistantMessage` containing the streamed type will be appended to the chat immediately. This allows the streamed type to be accessed and streamed from. For more information on streaming types, see [Streaming](streaming.md).

```python
from magentic import Chat, UserMessage, StreamedStr

chat = Chat(
    messages=[UserMessage("Tell me about the Golden Gate Bridge.")],
    output_types=[StreamedStr],
)
chat = chat.submit()
print(type(chat.last_message.content))
# <class 'magentic.streaming.StreamedStr'>

for chunk in chat.last_message.content:
    print(chunk, end="")
# (streamed) 'The Golden Gate Bridge is an iconic suspension bridge...
```

## Asyncio

The `Chat` class also support asynchronous usage through the following methods:

- `asubmit`: Asynchronously submit the chat to the LLM model.
- `aexec_function_call`: Asynchronously execute the function call in the chat. This is required to handle the `AsyncParallelFunctionCall` output type.

## Agent

A very basic form of an agent can be created by running a loop that submits the chat to the LLM and executes function calls until some stop condition is met.

```python
from magentic import Chat, FunctionCall, ParallelFunctionCall, UserMessage


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    # Pretend to query an API
    return {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }


chat = Chat(
    messages=[UserMessage("What's the weather like in Boston?")],
    functions=[get_current_weather],
    output_types=[FunctionCall, str],
).submit()
while isinstance(chat.last_message.content, FunctionCall | ParallelFunctionCall):
    chat = chat.exec_function_call().submit()
print(chat.last_message.content)
# 'The current weather in Boston is 72°F, with sunny and windy conditions.'
```
