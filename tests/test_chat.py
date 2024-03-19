import pytest

from magentic.chat import Chat
from magentic.chat_model.message import (
    AssistantMessage,
    FunctionResultMessage,
    UserMessage,
)
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)
from magentic.prompt_function import prompt
from magentic.streaming import async_iter


def test_chat_from_prompt():
    """Test creating a chat from a prompt function."""

    def plus(a: int, b: int) -> int:
        return a + b

    @prompt("What is {a} plus {b}?", functions=[plus])
    def add_text_numbers(a: str, b: str) -> int: ...

    chat = Chat.from_prompt(add_text_numbers, "one", "two")
    assert chat.messages == [UserMessage(content="What is one plus two?")]


def test_chat_add_message():
    chat1 = Chat()
    chat2 = chat1.add_message(UserMessage(content="Hello"))
    assert chat1.messages == []
    assert chat2.messages == [UserMessage(content="Hello")]


def test_chat_last_message():
    chat = Chat([UserMessage(content="one"), UserMessage(content="two")])
    assert chat.last_message == UserMessage(content="two")


@pytest.mark.openai
def test_chat_submit():
    chat1 = Chat(
        messages=[UserMessage(content="Hello")],
    )
    chat2 = chat1.submit()
    assert chat1.messages == [UserMessage(content="Hello")]
    assert chat2.messages[0] == UserMessage(content="Hello")
    assert isinstance(chat2.messages[1], AssistantMessage)


@pytest.mark.asyncio
@pytest.mark.openai
async def test_chat_asubmit():
    chat1 = Chat(
        messages=[UserMessage(content="Hello")],
    )
    chat2 = await chat1.asubmit()
    assert chat1.messages == [UserMessage(content="Hello")]
    assert chat2.messages[0] == UserMessage(content="Hello")
    assert isinstance(chat2.messages[1], AssistantMessage)


def test_exec_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    chat = Chat(
        messages=[
            UserMessage(content="What is 1 plus 2?"),
            AssistantMessage(content=FunctionCall(plus, 1, 2)),
        ],
        functions=[plus],
    )
    chat = chat.exec_function_call()
    assert len(chat.messages) == 3
    assert chat.messages[2] == FunctionResultMessage(3, FunctionCall(plus, 1, 2))


def test_exec_function_call_parallel_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    chat = Chat(
        messages=[
            UserMessage(content="What is 1 plus 2? And 3 plus 4?"),
            AssistantMessage(
                content=ParallelFunctionCall(
                    [FunctionCall(plus, 1, 2), FunctionCall(plus, 3, 4)]
                )
            ),
        ],
        functions=[plus],
    )
    chat = chat.exec_function_call()
    assert len(chat.messages) == 4
    assert chat.messages[2] == FunctionResultMessage(3, FunctionCall(plus, 1, 2))
    assert chat.messages[3] == FunctionResultMessage(7, FunctionCall(plus, 3, 4))


def test_exec_function_call_raises():
    def plus(a: int, b: int) -> int:
        return a + b

    chat = Chat(
        messages=[UserMessage(content="What is 1 plus 2?")],
        functions=[plus],
    )
    with pytest.raises(TypeError):
        chat = chat.exec_function_call()


@pytest.mark.asyncio
async def test_aexec_function_call_async_function():
    async def aplus(a: int, b: int) -> int:
        return a + b

    chat = Chat(
        messages=[
            UserMessage(content="What is 1 plus 2?"),
            AssistantMessage(content=FunctionCall(aplus, 1, 2)),
        ],
        functions=[aplus],
    )
    chat = await chat.aexec_function_call()
    assert len(chat.messages) == 3
    assert chat.messages[2] == FunctionResultMessage(3, FunctionCall(aplus, 1, 2))


@pytest.mark.asyncio
async def test_aexec_function_call_not_async_function():
    def plus(a: int, b: int) -> int:
        return a + b

    chat = Chat(
        messages=[
            UserMessage(content="What is 1 plus 2?"),
            AssistantMessage(content=FunctionCall(plus, 1, 2)),
        ],
        functions=[plus],
    )
    chat = await chat.aexec_function_call()
    assert len(chat.messages) == 3
    assert chat.messages[2] == FunctionResultMessage(3, FunctionCall(plus, 1, 2))


@pytest.mark.asyncio
async def test_aexec_function_call_async_parallel_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    chat = Chat(
        messages=[
            UserMessage(content="What is 1 plus 2? And 3 plus 4?"),
            AssistantMessage(
                content=AsyncParallelFunctionCall(
                    async_iter([FunctionCall(plus, 1, 2), FunctionCall(plus, 3, 4)])
                )
            ),
        ],
        functions=[plus],
    )
    chat = await chat.aexec_function_call()
    assert len(chat.messages) == 4
    assert chat.messages[2] == FunctionResultMessage(3, FunctionCall(plus, 1, 2))
    assert chat.messages[3] == FunctionResultMessage(7, FunctionCall(plus, 3, 4))


@pytest.mark.asyncio
async def test_aexec_function_call_raises():
    async def aplus(a: int, b: int) -> int:
        return a + b

    chat = Chat(
        messages=[UserMessage(content="What is 1 plus 2?")],
        functions=[aplus],
    )
    with pytest.raises(TypeError):
        chat = await chat.aexec_function_call()
