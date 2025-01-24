from typing import Annotated
from unittest.mock import ANY

import pytest
from inline_snapshot import snapshot
from pydantic import AfterValidator, BaseModel

from magentic._streamed_response import AsyncStreamedResponse, StreamedResponse
from magentic.chat_model.anthropic_chat_model import (
    AnthropicChatModel,
    async_message_to_anthropic_message,
    message_to_anthropic_message,
)
from magentic.chat_model.base import ToolSchemaParseError
from magentic.chat_model.message import (
    AssistantMessage,
    DocumentBytes,
    FunctionResultMessage,
    ImageBytes,
    Message,
    Usage,
    UserMessage,
)
from magentic.function_call import (
    AsyncParallelFunctionCall,
    FunctionCall,
    ParallelFunctionCall,
)
from magentic.streaming import AsyncStreamedStr, StreamedStr


def plus(a: int, b: int) -> int:
    return a + b


@pytest.mark.parametrize(
    ("message", "expected_anthropic_message"),
    [
        (UserMessage("Hello"), {"role": "user", "content": "Hello"}),
        (AssistantMessage("Hello"), {"role": "assistant", "content": "Hello"}),
        (
            AssistantMessage(42),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": ANY,
                        "name": "return_int",
                        "input": {"value": 42},
                    }
                ],
            },
        ),
        (
            AssistantMessage(FunctionCall(plus, 1, 2)),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": ANY,
                        "name": "plus",
                        "input": {"a": 1, "b": 2},
                    }
                ],
            },
        ),
        (
            AssistantMessage(
                ParallelFunctionCall(
                    [FunctionCall(plus, 1, 2), FunctionCall(plus, 3, 4)]
                )
            ),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": ANY,
                        "name": "plus",
                        "input": {"a": 1, "b": 2},
                    },
                    {
                        "type": "tool_use",
                        "id": ANY,
                        "name": "plus",
                        "input": {"a": 3, "b": 4},
                    },
                ],
            },
        ),
        (
            AssistantMessage(
                StreamedResponse([StreamedStr(["Hello"]), FunctionCall(plus, 1, 2)])
            ),
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {
                        "type": "tool_use",
                        "id": ANY,
                        "name": "plus",
                        "input": {"a": 1, "b": 2},
                    },
                ],
            },
        ),
        (
            FunctionResultMessage(3, FunctionCall(plus, 1, 2)),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": ANY,
                        "content": {"value": 3},
                    }
                ],
            },
        ),
    ],
)
def test_message_to_anthropic_message(message, expected_anthropic_message):
    assert message_to_anthropic_message(message) == expected_anthropic_message


async def test_async_message_to_anthropic_message():
    async def generate_async_streamed_response():
        async def async_string_generator():
            yield "Hello"
            yield "World"

        yield AsyncStreamedStr(async_string_generator())
        yield FunctionCall(plus, 1, 2)

    async_streamed_response = AsyncStreamedResponse(generate_async_streamed_response())
    message = AssistantMessage(async_streamed_response)
    assert await async_message_to_anthropic_message(message) == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "HelloWorld"},
            {
                "type": "tool_use",
                "id": ANY,
                "name": "plus",
                "input": {"a": 1, "b": 2},
            },
        ],
    }


def test_message_to_anthropic_message_user_image_document_bytes_pdf(document_bytes_pdf):
    image_message = UserMessage([DocumentBytes(document_bytes_pdf)])
    assert message_to_anthropic_message(image_message) == snapshot(
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "JVBERi0xLjQKJdPr6eEKMSAwIG9iago8PC9UaXRsZSAoVW50aXRsZWQgZG9jdW1lbnQpCi9Qcm9kdWNlciAoU2tpYS9QREYgbTEzMyBHb29nbGUgRG9jcyBSZW5kZXJlcik+PgplbmRvYmoKMyAwIG9iago8PC9jYSAxCi9CTSAvTm9ybWFsPj4KZW5kb2JqCjUgMCBvYmoKPDwvRmlsdGVyIC9GbGF0ZURlY29kZQovTGVuZ3RoIDI2Nz4+IHN0cmVhbQp4nKWS3UoDMRCF7/MU8wKdZn6SyYB4UdFeV/YN1BYEL6zvDyb704Cgu+BmScIc+M7JJASxjh3VqSjDy0f4DGhprC5rLRK08XyEaXO9hP1R4PIVmm6uQMQC17dwDqcfBOP2V0YcS5UxbTrjMIT9U0Uo5vYZDOdAPReqeSETh6FBdiTIZqJSYHiFuxjF7mF4DwU9cWZq1pOgh1mgRBStdOFhFCooZVKXm5DyKDwOf4cSqidzN18P918rzaiUmbdY6SZiKpjFhWSdmKbOKkYjt2S9geWXzs6nMhQuxP0qZtJatpJRRNU2ZBMZiY5mKcf6dhcrnqwoYqpP2rS3nP0WLntOXSBa0p3CNx0KoLkKZW5kc3RyZWFtCmVuZG9iagoyIDAgb2JqCjw8L1R5cGUgL1BhZ2UKL1Jlc291cmNlcyA8PC9Qcm9jU2V0IFsvUERGIC9UZXh0IC9JbWFnZUIgL0ltYWdlQyAvSW1hZ2VJXQovRXh0R1N0YXRlIDw8L0czIDMgMCBSPj4KL0ZvbnQgPDwvRjQgNCAwIFI+Pj4+Ci9NZWRpYUJveCBbMCAwIDU5NiA4NDJdCi9Db250ZW50cyA1IDAgUgovU3RydWN0UGFyZW50cyAwCi9UYWJzIC9TCi9QYXJlbnQgNiAwIFI+PgplbmRvYmoKNiAwIG9iago8PC9UeXBlIC9QYWdlcwovQ291bnQgMQovS2lkcyBbMiAwIFJdPj4KZW5kb2JqCjcgMCBvYmoKPDwvVHlwZSAvQ2F0YWxvZwovUGFnZXMgNiAwIFIKL1ZpZXdlclByZWZlcmVuY2VzIDw8L1R5cGUgL1ZpZXdlclByZWZlcmVuY2VzCi9EaXNwbGF5RG9jVGl0bGUgdHJ1ZT4+Pj4KZW5kb2JqCjggMCBvYmoKPDwvTGVuZ3RoMSAxOTg2MAovRmlsdGVyIC9GbGF0ZURlY29kZQovTGVuZ3RoIDEwMTU0Pj4gc3RyZWFtCnic7XwJeFRF9u+purf7dnd6uR06vSSd9O100gnpQCAJhEAkHZIAGvbNBBNJgEjYZAkoKEKYEYGIoowrOoI7rnRCxCbqkHEdF4Rx11FBxG0cBB1FUZP7TlV3JyHof5z3vve9b943dXN+dWo7VXXq1Km63Q1AAKAfggiDxpSVj6YmqgOgTswdN2bSxKlNI25LARCGYfqeMVOnjzL8SdoCQEKYHjRxak7uqtS9x7H+XEzXzigbXzlp24LvADJrAaw3zFlct5RuEXKw/FksXzHnkhXKne63vwSQ8gG00y5aOm/xK6urtgOYNmH64nl1jUvBCXqUX4T15XmLVl+UQ97IBij8Fgd5R8Pcxaueu+vLd5D/AUC3q6G+bu4R2/MojxzB+kMbMCM+T+/FNI4R0hoWr1iVu1LzJoDmDcyrXbRkTt2sLy/4BOfzPpa3Lq5btVSzx9SEZeMwrVxct7jeUTv4A1TGdZhXtnRJ4wo1C25CfhErX7q8fmn6W+P3A7h3AsT9CfME0AEFKxBVRZ7pshK+gSL4I0iYL0MOzEBpD2NdDaYF4EHNYDJ/IWB7aWTXBCiV4cfdP14m85wzQiXPMcBAfDR1y+tmgzJn9fJFoMxbXr8QlIb62ctBWVS34mJQemSCxrV90Y1jA7MsRd/pknQ8+66PM7JY/PKkEXt/3N05TwadEZP67h5ZbIzyFHuM5FnZXCEDHwL98SGQhw+BAnwIlOFDYCw+BCpgHOIMOB9REDaR61ADOs12TR6KSYrEwl/hIhqv09A4rUhZEKHPjMdPnDARgjjXFZrXuyaTPGkkaQ0yZauoVb/mCbYqIPIRAteuLboKNsyt5bGMOSJKKIPRMAmmQR3Uw3xYwST0yW2ARhT8cd/nrDXo0Y8BRkE1CHwFsiMrwHVv42MC7IPgyHp40ovH8S6sX451e2NULqtH0bIIJ5m3EmAoDIJEpk3Uex7qWwNWNRnKkMYijVPbYAbGuEZdPzBEHiDhXz55/8bzNXxNzv+F53TkoZf92iN6I4/Gzx5tvPb02Y8+UX+Qa1aCmWwVRfQDsACaojxBbV8S5SmYcbUivID2kRnlxV51NKgrc5TXIgdQAstx5etgEYzHFZ+BK74cGjFnCbDdMgStYTBquA5LWc4SWAGrYSnWUuBcWIz587DuxYgKDEDqkabAFKw1D1YiX4e5Z6Z66j2ANXOxh8H4KDiCBi777N5KMbUceYZ1mB8Z4UDe56Jof/OxhwYsa4z23shncwniXBio/RWT/XcCLYT2/2MhvxA0L8AdLBY/hl2/qf6MSP1YwHaD/t0+xUY4r0+6ohe/kbwAm36THIBi1Euaph1cSIma+8El+vHEAvUzpM9Z3DVf/ZyVs5j+HRuFowQ430fIfHgE9sPT5CS22g37oA3+Ag70QrfDGrgBNqKlzsSczbimU9CCy+AG4sJ9nQN3oiXfCQew7vmwFlfGTpzqF7AONgivY6sNYIJUtI5JaCnXkHHqSvRKh8Xfo48Yh5azlDSpleq16jb1HrgX9gl/UTshDnfHHHwOqF9p3lHfR4uuhhvhVjhMtukfwx11Pu67fcIf0aa2CzUiUeepP+IIvHApjkFEmz1AOmgApdfDZ8RJ1gilKOVuNaQ+i7XcUIO2uR3ayRAyhno11ep49QDYsY9VKPVWaIW9+IThKXiPGDUn1XvUk+CCbNxl61Afr5IOoatzfVcxW3zUUn8oxJIl8Cd4AQ4RH/kzXaIxanI1Qc1l6hvoMQfDdBzt/djyU/I9XYvPOuF5cbQ6Cvf8BrieaRueg49IIskhE8kM2p8uoXcIy9GzZvOdOBf30ma4BaV/SAJkLzXSg8Ld4kPiT9rkriOqGVfED7fhef5nYsKZKqSR/I68RT6mpXQWvY0eFW4QHxBfk+pw1heil7gGHoLvSTwZRiaTC0gDWUM2kuvJreQAOUQ+pyV0Gl1ITwgNwjLhKXEUPlPFRvH3mqs0V2s/76rserbrr13fq7nqVTAZ7WE9jv5GtP42tJOD8C4+h+Eo0ZA4YsZHIV4ynVyOz1pyDbmL7CIPkDbs5RA5Sr4g35DvyE8UHSXV0iTqpan4+Ohyeim9gd5OD+JziP6DnhYcQqoQEIYIRUKVsARHtVG4Dp/HhI/ERPGgqKKeczU3aXZodmke0jytOak1Sr/Tge6Vn+/uzOr8sAu6NnXd1NXa1aZ+hCeJC23KDR68/UxGv1WHvnsV3nPuRTt/nRhRd4kki4wk41Azs8gCsoysQk1eSbaTe/nYHyVPopbeJidwzCbq5mMeSIfQUXQiPhfSerqMXke30Tb6Fv1RkIQ4wSIkCFnCGKFGqBdWCKuFm4SQ8IrwgXBUOCX8jI8qGkSPmCr6xYA4RpwlrhTvED8TP9NUa17WfKI1aBdrr9KGtV9LQ6WR0iRpslQjbZX2Sm/oatE6n4HH4PHee58cEdYL5cJjcC3NE130Vfoq2vMsmCuMp2ipdBfZRK8gbTRNs0o7go4gE+Ck6EddP0930FN0hDCeVJCpsIAOjkjT2sQHMSoSn4Hj4pM4t1dR8iqtkaylJ7RGaCXM/wJ5ThgkBoSX4T3hMJHEO+FvooE4yHF6vzAJreApcaSmErzC7fCosIxcAY/RcryS/KTbgnY8gTyIfmEaySU/CHgzpRPQigqEj+H3sJC+A8dxH2+Cm8lccR5cC3lkDXwG9+Gu6K+5WJulTSAv0vliM+1H2oCKD7CzgKQRQWODK0mNsF17gr6Lp9tB0QAfCg/j6A/SR4Xx4knNFNKAO+AKuAqWqethtaZSfI3MA4HMgHTxCHq3NUKu6MV4HXqVavRpe3F3t6MfKBHGY44TLWcc2sV09BDb8bkF/YSIFjQf9/j56MVehTbtNBqGeRozQa+D3vjlrikwU70PblXnwcXqNhiA/mCjugYl7oJPYCvsIhu6LsdzNAV3zodknGY0PagZrQ6gzfRdOpXedOb6orbTiRP+js+jmBiJd8tm8W2YCsXqFvVNtO5M9LC3wmw8SY7hLL/CHsYKHZDXNYG2qKOFpTjfwzBZvV/1EAM0qItgIjwJ90oaqJMCuMYh8hrO93Kop1PUFUJ913zUw1bUQhC1tRL9z+Zg6fRpJcHikecUjRheOKxgSH5e7uBBOQMHZAey+mdm+NPTfKlexZOS7E5KdDkd9gRbv3irbDGbjHEGvU7SakSBEsgu942uVUL+2pDo940dO4ClfXWYUdcrozakYNboM+uElFpeTTmzZhBrXtSnZjBSM9hdk8hKERQNyFbKfUroQJlPCZOZkyuRv6bMV6WEjnN+POev47wJea8XGyjlzoYyJURqlfLQ6Esamstry1BcS5yh1FdabxiQDS2GOGTjkAs5fEtbiGMk4Qx1lA9vwRuyCQcVSvSVlYdcvjI2gpCQXl43NzRpcmV5WZLXWzUgO0RK5/hmh8A3KmQJ8CpQyrsJaUtDEu9Gmc9mA1crLdkdzVvCMsyuDRjn+ubWVVeGhLoq1oc1gP2WhRyXHXP2JFF4fGnlxt6lSUJzuXO+wpLNzRuV0M7Jlb1LvQyrqlAGtqXpo2ubR2PXW1CJFVMV7I1uqKoMkQ3YpcJmwmYVmV+9r5zl1C5QQnrfKF9D84JaXJrE5hBMWe1tTUwM7lOPQGK50jyt0ucNFSf5qurK3C02aJ6yeo8rqLjOLBmQ3SJbI4ptMVuijNHUm6nvLuMcr864iindmiVsRL5z0SBCyhwFR1LpwzkNY1A/DJrnDMNqGKoItgrNxRWZH9KX1jbLw1k+ax/SpMs+pfk7QAvwHf/HmTl10RxtuvwdMJbZSbepYXmMDwUCoawsZiJSKa4pjnEkTw8ZkH1JmPp8S2UFI1QfTELd1lUNz0H1e71sga8OB2E2JkJNkysjaQVmJ7VCMCdQFaK1rKQjVpIwnZU0xUq6m9f60JLb+I07IaTzd/9ZZHu/8obhIWL/H4rrI+UVU30Vk2dWKuXNtVHdVkw7IxUpH9ZdFuVC/UorhSQa5WiSwEvRKKu7K7NEpTEkpuOflhv13LCkQ6vkOUQZHZJrx0awyuD1/sZGYfUka8WjnmbRYYaGB85MjzgjfcbwjM0CDhiPyoppM5ubDWeUoalFOjw3GqHFw7RKr1Iagum4M9PxL6x2DGNUlRQKospKWQW0v0hWNHlGxaQoX4WBWeeA7NHo6JqbR/uU0c21zXVhtWm2T5F9zfvo0/Tp5qXltTHDCavtVyeFRm+pQl01kOG4KSiMavGRTZNbgmTT1JmV+2QAZdO0ylZKaGntqKqWNCyr3KcABHkuZbkskyUUloAKgpNspTpeP2lfEKCJl4o8g6fnhAnwPF0sj8CcMI3kybE8inliJC/I81hgPqZ0WmVv6+FbsmoAP/DYZ3zVcTpd39cc4RdffiS8CceCNkp9M7S0JwelGPX63yZbj9JjQcf7OqNjTlJPU8aaDIa+UsRfka3rxf9r2SjFEhf322THQc8EDVH53UEXIV2PbB1KkY1G6BN+WbYReiYY11e2PkK6nqaMtZpMfaVo/rdl68+U3c9s/m2yzdAzQRMAnDFdQ6RDQ49sA0pJkOW+UrR9M3iwQM8EzX1lGyMU1zOsOJTijI/vK0Xqm8GDFaXHghyV3x1METL3yDailCSbra+UszYTDzaU3tNPRH53sETI0jNlM0pJttt/m+wE6Jlgvyh1BzlCcs+ULShFcTr7Sjlro/LggJ4JJvSVbY1QfI9sGaV4Xa7fJtvZS7Y9Kv9/kG1FKelud18pZzkBHpJw5LHg4vPoFWwRSuiRHY9SshSlr5SzNioPKfiWGwtsOIm9Cx0RcvQMy4ZSBvp8faWctVF58ELPBD1R+d0hMUKJPbIdKCXX7+8r5ayNykMalxjrB3qloh0huXum7EIpQ/v37yvF0jeDhwwuMdYP9EoBUxin5J4pJ6GU4dnZfaWc5QR4yIIe5WVE5XcHJUJKj+xklFKam9tXSr++GTwMgp4JsjPxjOmmRyi9Z8qpKKVi2LC+UhL6ZvAwFAZ287lR+d0hK0JZPW7Bj1KmjhzZV4qjbwYPRZDXzRcg5fcuHBihgT1uIQulVJeX95WS2DeDh1HQM8FzkAp7F+ZFKK9n6w5EKXMrKqBPOGuj8jCWS4yEUqQzpjssQsN6/F4uStkH04TMPX6n59CTQn84gkSF/q2BZM8+IUNIbh3hCYYF3574hFxLyQBBwbtRDkcFcQnSbqT9AvuOaJaQwr5LQVyH1IS0G2k/0iEkdL6IrFRBWoK0A+kIKxGSBXer4pFLMgQXtnXhXcsiOOAEkookgAcxB2ki0iykrUg7kLS8HstZgrQOaT/SSV4SFByt2/Jw7I7Wq3m0Z8GiXJ6siySra3hyz/lVkXj85Ehcdm6k2vBItcH5keyBoyJxRnYkjk/PbWKxwZTbUWIX7DhJOw58KSKhz4KFEHQAO4UECCFRQRvNCQrxe9L8uTv2CyIQgQoE5oJH7RBIq8maW2KgKj2Bx42HfkWPR0ro8T1ma+6OkvPoUdiNtB9JoEfx+Yh+BOvoEaZzxGKkHUj7kQ4inUDS0iP4HMbnQ/ohWOgHkINUjDQLaQfSfqQTSBL9AFGm77P3KI6ML0ai9H1Emf4Np/U3RAt9D7n36Hs4tNdbCwpz93EmkBNlPOlRxpEUZeLtuWH6Wuvp/mhRflxptKgnhFQ0zTwhtTV9sCcsOFuL5nvC9OM9SsCzs2QQfQNCSHjVRZSRFKRJSLVIS5G0yL2F3FvQhHQd0k6kEBJaGaKMpNCXkF5BegsGIQWRJiHp6KFW7CZMD7b6R3lK7PRV+gI6AQ89QP/C41fo8zx+mT7H4xcxTsH4Jfp8a4oHSuKwHLCNjLGMcQ6Wa+if96TFe9QSK92PuvMg5iAVI01EmoW0FUlL99PU1rmeeBTyBLyEFwUPbYUveHwf3KWD4AJP0F+KBqgw8A8/BzmEHcoOPw36b7oVkwz8125DjoH/yi3IMfBfth45Bv5FlyDHwD93AXIM/DNnIcfAP3EacghhesfjaRmegokLiVJioZeili5FLV2KWroURHope+C0yMZ2W2tWFmpsezDQP8vT1E6aniRNU0jTXaSpnjStJU3rSVMRabqQNAVIk5s0pZCmIGl6ggxDVTSRYNsZycKgkzS9RJoeIU2NpMlPmtJJUxppUkhBMEy9refm8aicR3tK2KbD+JyR6H0s1Isa9aLNe9En7Ec8iKTyVBArKamRyq4UFqfuySqOpAcOz11SMpY+gw2fwWV4Bg4jibhAz6AZPYNCnkEBFsRipFlIHUgnkFQkLdZOxYFv5WhBzEEqRpqFtA7pBJKWD+cEEoUl0SHu5gPLiQ56IkvRZ/BhXxR4qTeYLLvlgDxW2OomlhQyMUVNoQXA76TxVp01TEx7vzf98L0J9CV6ei3dCsm4ENdF462tp5M9YXJLq/8JT0kCuRlSRLQ6Ugh+ko7xMGjk6SHg1rE4H9z0IYxzW90zsJml1Z/taSdm1mqv57T7mOcLd5gi+7n7Cc/bSlgkrZ43MeehvZ433Js9L+aEdZjzpD9MMGpXeNV97mGeR17iVddjwfZWz1oW7fVc4R7jWejmBfWRggsbMRW0eKb4Z3rGorwy92xPsBFl7vUUuy/0FEVqDWFt9noG4RACETYLB9vfzTv1pXCB0wvCpCGYLd0kVUoTpaFSrpQteSWPlCwlSTZdvE7WmXVGnUGn02l1oo7qQGcLq0eCAfZ9v03Lf+yhFfmX/5yXKfCfD/CfBFCio3AehPoJFbRi6ihSEeqYAxWzldCpqb4wMUyeGdL4RpFQfAVUTBsVGhaoCEvqlFBBoCIkTbqgsoWQa6swN0Q3hQlMqwwTlWVtSGKfX+4DQqwbrkliceaGa6qqwGm/pNhZHD/SWji67BegNoqBnuA8g08O3VQxtTL0YHJVKJcxanJVRegP7APOfeQbcrK8bB/5mkVVlfuEkeSb8iksXxhZVlVVESYzeD1QyNdYDy3ma15PhwczqweKLiVSb3ukXjq2x3ppLMJ6ej2k83rpej2vJxJWr6UxrbysJS2N13Eo0MjrNDqU3nVeSsc66em8jr0JXuJ1XrI3sTqhkbyK241VUty8CkkEN6/iJom8yoyeKjnRKpu7q2zmPQmkp447Usd0JFbHdATrBH5rqB8VCJA9I6rmVLMPh2t95fVItaGrL2lwhppmK0rLnKrop8b+2tlzGlhcVx+q8tWXheb4ypSWEdW/UFzNikf4ylrwvjitsqU6WF/WOiI4otxXV1a1Z8yk/IIz+trc3Vf+pF8QNokJy2d9jSn4heICVjyG9VXA+ipgfY0JjuF9AbfxSZUtOhhVVVodiffQOAPaa22St2qUXV46khvvCK9zbVI73lZ2QVygKmT0jQqZkFjRgJIBJawI9xQrMrNvAKJFzrUjvEntZFe0SMZsq28UBFasbFwJzvL5ZZG/RgyYtWIlU3gEA42/FrCsPBSsK2tcgW8JoaypFaHiyTMrWyQJc2vZlELDY3lxceVhtSOSORAzh7NMQeiuyPKKWJ5eH6149vqvjMalbBc00Sf2kGAKWQGNVUIopWIaRVcwLfpRazvepdjx0FiFE2wkAdIYkxEddiAAkTSwOcdoxcooF9XFimgcaYlNGmMq6Q5MWYFuja1AgSwIIBAWNIJAKF4znZp/xHXADzoV0AWqXaAHvdqJb/IG/ruEOEQjGBFNYEI0c7SAGVEGC6IV8We8hloR+0E8og36ISYg/gR2sCE6IAHRifgjuMCBfCK4kE+CREQ3x2RIQkwBt3oar74MFUhG9OLF9jSkgoLoQ/wB3zm9iOmQiuhH/B7fRH2ImZCG2B/8iFkcA5ChnoJsyEQcwHEgZCHmQABxEAxAHIz4Hb4TDkTMgxzEfBikfgtDOA6FwYgFkIc4DPLVf+IbGMPhMARxBMciGIp4DhQgjoRhiMVQqH4DQRiOWAIjEEdBEWIp4tdQBucglsNIxNFQrJ6EMRBEHAsliOfCKMTzOFZAKeI4KEMcD6PVEzCB40QYgzgJxiJOhnPVr2AKx6lwHuI0qFCPw3QYjziD4/kwAbESJqr/gCqYhDgT8ThcAJORr4apiDUwDfFCjrNguvol1MIMxDo4H3E24t9hDlQhzoWZiPVwAeJFUK1+AfM4NkAN4ny4UP0cFkAt8gs5LoI6xMUwG/MvhjmISzguhbnqZ7AM6hGXwzzERo4roEH9FFbCfMRLYAHipYifwCpYiLgaFiNeBhcjXs5xDSxBvAKWIq6FZeoxWMexCRoR18MKxN/BSpV9334J4pUcN8Cl6lG4ClYhboTViJvgMsTNcLn6ETTDGsSr4QrM2YL4EVwDaxGvhXWIW2E94nWIR+B6+B3iNvg94h/gSvUw3MDxRtiAeBNsRLwZNmHpLYiH4VbYjLgdmtUP4Ta4GvF22IL4R453wLWIO2Ar4k64DvFOxA/gLrge8W7YhngP/AHxXrhBfR/ugxvVv8H9cBPiLrgZ8QGOD8ItiA/BrYgPw22Ij3B8FG5H3A1/RAzBHYgtiO9BK+xA3AM7EdvgLvVdeAzuVt+BvRwfh3sQw3Av4j64D7Gd4xOwC/FJeEB9G56CBxH/xHE/PITYAQ8j/hkeQXwaHkV8Bnarb8GzEEJ8DlrUN+F5ji9AK+JfYI/6BrwIbYgvwWOIL8NexFfgccQDEEZ8FfYhHuR4CNoR/wpPIr4GT6mvw+uIr8Eb8CfEN2E/4lvQof4V3ub4DjyN+C48g/gePIv4N47vw3OIH8DziB/CC+ohOMzxCLyoHoSP4CXEo/Ay4sccj8EriJ/AAcRP4VXEz+CQ+ip8zvEL+Cvi3+E19QB8Ca8j/oPjcXgD8St4S30FTsDbiCc5fg3vIH4D7yL+E95D/Jbjd/C++jKcgg8Qv4cPEX9AfAlOw2HEH+EI4k/wEeLPHDvhY/VF6IJjiCp8gvhfn/5/36d//R/u07/8zT79i1/x6V+c5dM//xWf/tlZPv3T3+DTj3X79OVn+PSPf8Wnf8x9+sdn+fSj3Kcf7eXTj3KffpT79KO9fPpHZ/n0I9ynH+E+/ch/oE9/9/+RT3/jvz79vz79P86n/6ff0/9zffqv3dP/69P/69N/2af/5f8Dn075PwpkP+AR2K9hvFavNR2B/bOznxWh4+egBgUoYgf76LkdYSPqWID0oJMWgYEWzcJtug5NRdyJ5TvFO29xBuRTNTXHofj44EF5Q/IS2g8cOMD+Td4d2MtMTTueCsnEFYxXPKRU505OoYRa5RQL6BxhtavNaBw1HZlv2uLiOHMqmGo0IedX9MQTNJnodL0iy4gGiwXRyXPC6rfBDKNRO12f6EmWzaypbDBhM9nI5MkK4d8Cs3ZoHafaWFPOsNbI/Ijdcub7NiYFmR+C2B65mpQR1Ww+kU+0aoo6EYuiyZrj7EOy4qLOIkaDB5WuDg4VkiSdVqfRiTpR63ImOqk2zmA0mAyCNsFus/ezC9okweEl8WYEp87tJXaD1QuBAAkEsjCsJzV5Vm+uw+6wxyfYqJn60r25QwuGDh2S78/w+7x3kNMPzVxbtaJxwmXXH9jQ1UIKr793cPn4mxdNeKTrFU17QvK42V0Hn72/q+uButxHhg4u/+K+T7/PYt907wIQN6Du9XBX8BytJkWn2yoRSQJBZPoHnXS7QpU4ShPjRD1lOtNzLeoNTPn6f6m/YBxXoDEujmuvK6bGkzE1GpgaJ8jfBmKKHM81OQEtZfwxKGYKjC/MqSmSmSJRBQleTruED37+hIY6J2naH+ka/kjnRVE7EjtxLia8KZwMptRbF9pohVxhu0C+wCbGGVMsZjM4nJF5xcdsKj5mU8j8sJfZRrxfx8zLynidgc1EJ7Px65g1DWAT0CUqiQT/Ep0mrhMT14mJ68T07+rkx7N04uptWjHbmiAvq+FZ4yPWFdUMNzBSAxHzSKFoHF6vFfluy6D9t41ftK3qq64XuzaRy5+8o2bc4Cu7NmvazfH1exc/0dXZ+bBAtqyr/n2Cie3jQajDdtShBB8GXVo+O4mjVmJzlDiSsHqa6wxHfTpoZZxWwyYvccTin9oYg8U/Ba2M09AUUWDmpNWI+jBt3KOIRAwT8rhWITRHIALyjxGuubD6eTCOa08XVd03sV14NKbDn2M662rjKmMSdXtv7b0h5W+Z0Ryr+VQukosi6kJF9XwIjXY0hFkS7deVLDZ3JWlMjzzy4z/Ris5TPxfd4ki8DRbQXcFsvUmf5TIlZvU3ZWUVmoYmFCQNzzo3q8ZUk7XAND+rdlCz6ar+2+23JT5gSsjEobexVc1gc3Ax7j7Xg5l7XU9kPus6mPlawgeZujI7SWF2ZGUTiI9nqDEyHMK+5JvIOI/D4wxkZ+UXioXZ54pjs2foqgIX6eYHLjFuNL5oPG06HbAW5JuJKOek5TtyvTbnrP5L+tP+7hxzsXmreYdZNWt2mHebT5gFM7c2tjxmI9Mlpv/exlRrZkPwyrIWC5gCzVqLBdHPTN7sZMttNrsFR5g+GDQ5s5kA5402t1uC7qFDeYYh1y3E9a+T67h1R83hB76p2BIFzUwaaPmSpXvTwuo/eN+MCcax3DSRrSumj2FrznzL1YfM+8wEkOPjSosteFqYXhA0ZwTBL/sV/yD/br+mMKx2tJnNdLo/rL4VY77l+9g/mBUGTSm+/EGFHYV0ZyEpdLAJLGSiHTp+oqQ7U3N0rHYOPxFy+E7OSduvPailHm2xlmpt3LxtrI6Wt9GamS61RjYFrZNNQWtk49fyHa01M4VqZTZe7eBh3duYGeUytnW/RZBrlgVO8ZMi5vgCfJcHPvkET8fiY4Hi452BY1bc373aLsM0/hUSa7yjcPAgqGH5ZBlGsCxdq/Wl+ofkD8UzgT1D8nHvp2qljJE0L9eOp0ZCgs3u8PkFrWRGH2HPY4fHEKFo7r4Fu58c0zh2yML35pG88k3rVieHnBcf2rzpwUmy3pH6pNsx+9kl1bmL5zfc5U/+/fTRD22YsH6CzWxKTEs3XDzgnKplzmVXVwTrzhu46uRPG84ZRj7IdMuZ43PG1l4w8ZxLmU+uwN2UgrspAc/2O4IOfDFLoNOFGk2NfnpcvbBQs0RfH6dLYCbA1GZFJjiFccluhhnx72p+tJ1KFAfHD3cNdpfEj08scU+Or3ZNcdfFL06sc6/Srko4RU85ZbATi8nhmGSvtS+1C3a35Tp5p0xlWUxyGyRopw8CQUNh9keYUZiZCciEkBv7ucU4tIqTZ10wIoeBI2hCa+R+B5mv+ChNbBux9TUxUfqMrPyQiZgSPZjak+7PZ/HjzOg8xGNnG6GaCbLnydx6orcQbnNymhRMy8r3SMXSREmQuAVKRlYiKcyiJG5dkpv1LpmZdUlu1q9kZ8OQXCn5Bb0Pz0BNgJ+exzBvWSBwalmv4+J4J9rJseLj/DDtXFZEmCXFF0bODm5Hy4mDGRFYZcjLBatN8tqZmRCvn5uScGF79lf7vug6QWzvv0nM5OfPDa0b5mzpfI9ONg6bsXnNA2SG4+424iECMZLMrg+7TsvK7vYGcuNVpQ33sZNlI14j2b95tpEb94EdVZXgyBeY0+MbJl0cIpQL7SaRZyU4XPkOndVotQkaAha3RrLhdYl5F6YcY2yVjGw3ZzGtGdP1wbyh+aqedOiJPciUZw/y62AmRxvzDHrmeaz8Ysg9jz6R1dOzwz6OKVdvY1ahZ5e8ODYmdpXk6VN7+R1ygp2td//8ofkh+0k7XWrfaQ/ZVbtopza+nDa+gDa+wLZ0dpJ1BGUc1Un20wwF2AuByA/86N3gx6CDDQv4YEDHBgNi7FYQtLPOgXJnS/kpNyFhzKReFwP0CYHovRO5biOIlkTvoOhNrIXWQsKWGu+hZq1ZSjdrjUnEpLMkEWD3y/WAhkMCedY861DmLBKsPms+W3JtgnVj29qOSx6taFu5cNI1RZr2zm+21dxze+cseufGy6dee0XnE7jDN7H/hwXXFd8RyP1BFzWw6QsctRwlQ/TO8HPskIgwmhgjsk2SzDgaxzQocNRylDhi487YwRJhNDEGG3cGkxlHRbYIAkctR4kj79lkivbMGE2M4T0PZ5x+KFP/RP11+p36kL5Df1h/Ui+B3qNfqm/S74hmHdGreoNHT4BIIhX0WoGt8ADe61oCWo1WNGildA2IO8SdYkjsEI+I2g7xpEhBVMRDmBJFZsNs+cXu5Rf58osG1r9oY8svRo5pzkQuOHycBmYK4gRdXyNYjlcbdsEpPh7gu5oR29fLlwV+LfTDty8B13tTW1ub+OXBgz8liP6f3mO7tFj9XGhBfz1IaAn2c3BTdnJ0ccxkZzRTdkaM8ceY9BiTFmN8MSY1xnhjjMJ27jq+fqm21OH68/RlaTNS61PX6K/VX5l2X7+Hsp8WTHpHotMxqCL7LYcmiU6nVM4lBme1rlpfbaiOqzZWmxboFugXGBbELTAuMLX52zIsGf60jLT+Q9NmGqri5vrnZq7wrUhrSvuD4Xbjtsybs28cdI/hAePdGfdk7vE/57fzuTDtpsYYX4xJizHR+WpjU9DGJqWNTROPqrD6YTA+pXCmLiPdaBATFX+CGDcwOZHdnlJd2WxdPa5i10TXLNdu10GX1uLyuJa4DrtEj2uri7qewl2fgEclP6GCNlZdJkFCZXIIX1SITCg7sfbY7Pn85JLN1nxCBlYnL0qmye4ESYzcmrhFfRqzmk+D/Zj3EN0D4zz4mpLmCvZz5uey5kPY3cXljCBzhS47Mz+Xwlq6FNbKxe8uLn7GsFJc+3Z6AUjqN3v5dk7LQkGPuQsPZZEs1idrn8WuvkwoZ1j7LGbETEQWu48xKVmJfARePC9rcztyaXFuUy7NZYdwGvChgMzdnBJRPuVGwmfErcXDxqZwK1TSLDKbsoWP3aKwyha2ofxsCBYz69/CL7YWLevZknoYSDFMRG/lGhw9M2uWjY+5TOYqAzLGyyfELmaBwDJ2cvZyqseX4yGKcfHxZfxahvsucEzu5FHkYha9l6GXDWYMSPFpbNl+qxwv95MFbapJSQJ9ppRENAMQUmyY9Jp9SZDqMxl1/Q1JJDNDb9AGxCTwyMnMLwfYa0sE+EtLVmD9+vXQy8Gz/V3Tk8Eq9SvgJza7+2UMpHgZLBjKHbpD8jN3nmBjHx/wd0R+WyxutWy+fM2qIel/eP7WiSXDsq6fesVTM60hY+P8NQvs9pykK/ffPGP+81ccfJec4164vL7sHJ8zPffc9RPGrM70BMZePs85pXpKgc+d3M+QlleypnrmjvMfZh4kTf2GZmluBQfx7ANj9C07Lva6rYsxUozRxhgDM3OfP1/PrGQqMk0u9LRGk4EIYJf1AYtBa8c3DoucCqnEFM+P3HhuD/EG/iafbiSqpCvXl9dKS6Um6TpJBEmRdkohqUM6JGkldn9jvldidsUsRWIXCeaDpcjVIsrwS1fknNYy5mQwjtmepOV3L2bg/PrVTheAkwxtuai3O8aV+faYfDx6Lh/7tojd59FBW/GyZc3Lk19knjlaNd3BlsE/xOobkmctsOYl+Kw2toJUThxXNHtR9pVX7nnssX6BzJQ7d8gj6++ic7YQaVHXNVs6/zA+OxHgfwEo7oHACmVuZHN0cmVhbQplbmRvYmoKOSAwIG9iago8PC9UeXBlIC9Gb250RGVzY3JpcHRvcgovRm9udE5hbWUgL0FBQUFBQStBcmlhbE1UCi9GbGFncyA0Ci9Bc2NlbnQgOTA1LjI3MzQ0Ci9EZXNjZW50IC0yMTEuOTE0MDYKL1N0ZW1WIDQ1Ljg5ODQzOAovQ2FwSGVpZ2h0IDcxNS44MjAzMQovSXRhbGljQW5nbGUgMAovRm9udEJCb3ggWy02NjQuNTUwNzggLTMyNC43MDcwMyAyMDAwIDEwMDUuODU5MzhdCi9Gb250RmlsZTIgOCAwIFI+PgplbmRvYmoKMTAgMCBvYmoKPDwvVHlwZSAvRm9udAovRm9udERlc2NyaXB0b3IgOSAwIFIKL0Jhc2VGb250IC9BQUFBQUErQXJpYWxNVAovU3VidHlwZSAvQ0lERm9udFR5cGUyCi9DSURUb0dJRE1hcCAvSWRlbnRpdHkKL0NJRFN5c3RlbUluZm8gPDwvUmVnaXN0cnkgKEFkb2JlKQovT3JkZXJpbmcgKElkZW50aXR5KQovU3VwcGxlbWVudCAwPj4KL1cgWzAgWzc1MF0gMTcgWzI3Ny44MzIwM10gMzkgWzcyMi4xNjc5NyAwIDYxMC44Mzk4NF0gNTEgWzY2Ni45OTIxOSAwIDAgMCA2MTAuODM5ODRdIDY4IDc1IDU1Ni4xNTIzNCA3NiBbMjIyLjE2Nzk3XSA4NyBbMjc3LjgzMjAzXV0KL0RXIDUwMD4+CmVuZG9iagoxMSAwIG9iago8PC9GaWx0ZXIgL0ZsYXRlRGVjb2RlCi9MZW5ndGggMjg0Pj4gc3RyZWFtCnicXZHdSsQwEIXv8xRzuV4s/W9XKAWtCr3wB6sP0CbTGrBpSLMXfXuTzLqCgQQ+5pyTySRqu4dOSQvRm1l5jxYmqYTBbT0bjjDiLBVLUhCS2wuFky+DZpEz9/tmcenUtLK6BojeXXWzZofDnVhHvGHRqxFopJrh8Nn2jvuz1t+4oLIQs6YBgZNLeh70y7AgRMF27ISrS7sfnedP8bFrhDRwQt3wVeCmB45mUDOyOnargfrJrYahEv/qFbnGiX8NxquTxKnjOH1sPKVVoDwnuiUqA2VZoCImImVBSmfwVCZEJ6IidHC5K/29+dpofk/xLalPwVuUlEvxVXaJIJN/j5/7dVj8bIybU/icMCA/Gqnw+n961d7l9w8dopEJCmVuZHN0cmVhbQplbmRvYmoKNCAwIG9iago8PC9UeXBlIC9Gb250Ci9TdWJ0eXBlIC9UeXBlMAovQmFzZUZvbnQgL0FBQUFBQStBcmlhbE1UCi9FbmNvZGluZyAvSWRlbnRpdHktSAovRGVzY2VuZGFudEZvbnRzIFsxMCAwIFJdCi9Ub1VuaWNvZGUgMTEgMCBSPj4KZW5kb2JqCnhyZWYKMCAxMgowMDAwMDAwMDAwIDY1NTM1IGYgCjAwMDAwMDAwMTUgMDAwMDAgbiAKMDAwMDAwMDQ4MiAwMDAwMCBuIAowMDAwMDAwMTA4IDAwMDAwIG4gCjAwMDAwMTIwMzkgMDAwMDAgbiAKMDAwMDAwMDE0NSAwMDAwMCBuIAowMDAwMDAwNjk5IDAwMDAwIG4gCjAwMDAwMDA3NTQgMDAwMDAgbiAKMDAwMDAwMDg3MSAwMDAwMCBuIAowMDAwMDExMTEyIDAwMDAwIG4gCjAwMDAwMTEzNDYgMDAwMDAgbiAKMDAwMDAxMTY4NCAwMDAwMCBuIAp0cmFpbGVyCjw8L1NpemUgMTIKL1Jvb3QgNyAwIFIKL0luZm8gMSAwIFI+PgpzdGFydHhyZWYKMTIxNzgKJSVFT0YK",
                    },
                }
            ],
        }
    )


def test_message_to_anthropic_message_user_image_message_bytes_jpg(image_bytes_jpg):
    image_message = UserMessage([ImageBytes(image_bytes_jpg)])
    assert message_to_anthropic_message(image_message) == snapshot(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "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",
                    },
                }
            ],
        }
    )


def test_message_to_anthropic_message_user_image_message_bytes_png(image_bytes_png):
    image_message = UserMessage([ImageBytes(image_bytes_png)])
    assert message_to_anthropic_message(image_message) == snapshot(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "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",
                    },
                }
            ],
        }
    )


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True", [bool], bool),
        ("Return the numbers 1 to 5", [list[int]], list),
        ("List three fruits", [list[str]], list),
    ],
)
@pytest.mark.anthropic
def test_anthropic_chat_model_complete(prompt, output_types, expected_output_type):
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = chat_model.complete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_usage():
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = chat_model.complete(
        messages=[UserMessage("Say hello!")], output_types=[StreamedStr]
    )
    str(message.content)  # Finish the stream
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_usage_structured_output():
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = chat_model.complete(
        messages=[UserMessage("Count to 5")], output_types=[list[int]]
    )
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_no_structured_output_error():
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    # Should not raise StructuredOutputError because forced to make tool call
    message: Message[int | bool] = chat_model.complete(
        messages=[
            UserMessage("Tell me a short joke. Return a string, not a tool call.")
        ],
        output_types=[int, bool],
    )
    assert isinstance(message.content, int | bool)


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_raises_tool_schema_parse_error():
    def raise_error(v):
        raise ValueError(v)

    class Test(BaseModel):
        value: Annotated[int, AfterValidator(raise_error)]

    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    with pytest.raises(ToolSchemaParseError):
        chat_model.complete(
            messages=[UserMessage("Return a test value of 42.")],
            output_types=[Test],
        )


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = chat_model.complete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_parallel_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    def minus(a: int, b: int) -> int:
        return a - b

    chat_model = AnthropicChatModel("claude-3-opus-20240229")
    message = chat_model.complete(
        messages=[
            UserMessage(
                "Use the plus tool to sum 1 and 2."
                " Use the minus tool to subtract 1 from 2."
                " Use both tools at the same time."
            )
        ],
        functions=[plus, minus],
        output_types=[ParallelFunctionCall[int]],
    )
    assert isinstance(message.content, ParallelFunctionCall)
    assert len(list(message.content)) == 2


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_streamed_response():
    def get_weather(location: str) -> None:
        """Get the weather for a location."""

    chat_model = AnthropicChatModel("claude-3-opus-20240229")
    message = chat_model.complete(
        messages=[UserMessage("Pick a random city. Then get its weather.")],
        functions=[get_weather],
        output_types=[StreamedResponse],
    )
    assert isinstance(message.content, StreamedResponse)
    response_items = list(message.content)
    assert len(response_items) == 2
    streamed_str, function_call = response_items
    assert isinstance(streamed_str, StreamedStr)
    assert len(streamed_str.to_string()) > 1  # Check StreamedStr was cached
    assert isinstance(function_call, FunctionCall)
    assert function_call() is None  # Check FunctionCall is successfully called


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_document_bytes(document_bytes_pdf):
    chat_model = AnthropicChatModel("claude-3-5-sonnet-20241022")
    message = chat_model.complete(
        messages=[
            UserMessage(("Repeat the PDF text.", DocumentBytes(document_bytes_pdf)))
        ]
    )
    assert isinstance(message.content, str)
    print(message.content)
    assert message.content == "This is a test PDF."


@pytest.mark.anthropic
def test_anthropic_chat_model_complete_image_bytes(image_bytes_jpg):
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = chat_model.complete(
        messages=[
            UserMessage(
                ("Describe this image in one word.", ImageBytes(image_bytes_jpg))
            )
        ]
    )
    assert isinstance(message.content, str)


@pytest.mark.parametrize(
    ("prompt", "output_types", "expected_output_type"),
    [
        ("Say hello!", [str], str),
        ("Return True", [bool], bool),
        ("Return the numbers 1 to 5", [list[int]], list),
        ("List three fruits", [list[str]], list),
    ],
)
@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete(
    prompt, output_types, expected_output_type
):
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = await chat_model.acomplete(
        messages=[UserMessage(prompt)], output_types=output_types
    )
    assert isinstance(message.content, expected_output_type)


@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete_usage():
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = await chat_model.acomplete(
        messages=[UserMessage("Say hello!")], output_types=[AsyncStreamedStr]
    )
    await message.content.to_string()  # Finish the stream
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete_usage_structured_output():
    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = await chat_model.acomplete(
        messages=[UserMessage("Count to 5")], output_types=[list[int]]
    )
    assert isinstance(message.usage, Usage)
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0


@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete_raises_tool_schema_parse_error():
    def raise_error(v):
        raise ValueError(v)

    class Test(BaseModel):
        value: Annotated[int, AfterValidator(raise_error)]

    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    with pytest.raises(ToolSchemaParseError):
        await chat_model.acomplete(
            messages=[UserMessage("Return a test value of 42.")],
            output_types=[Test],
        )


@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = AnthropicChatModel("claude-3-haiku-20240307")
    message = await chat_model.acomplete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)


@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete_async_parallel_function_call():
    def plus(a: int, b: int) -> int:
        return a + b

    def minus(a: int, b: int) -> int:
        return a - b

    chat_model = AnthropicChatModel("claude-3-opus-20240229")
    message = await chat_model.acomplete(
        messages=[
            UserMessage(
                "Use the plus tool to sum 1 and 2."
                " Use the minus tool to subtract 1 from 2."
                " Use both tools at the same time."
            )
        ],
        functions=[plus, minus],
        output_types=[AsyncParallelFunctionCall[int]],
    )
    assert isinstance(message.content, AsyncParallelFunctionCall)
    assert len([x async for x in message.content]) == 2


@pytest.mark.anthropic
async def test_anthropic_chat_model_acomplete_async_streamed_response():
    def get_weather(location: str) -> None:
        """Get the weather for a location."""

    chat_model = AnthropicChatModel("claude-3-opus-20240229")
    message = await chat_model.acomplete(
        messages=[UserMessage("Pick a random city. Then get its weather.")],
        functions=[get_weather],
        output_types=[AsyncStreamedResponse],
    )
    assert isinstance(message.content, AsyncStreamedResponse)
    response_items = [x async for x in message.content]
    assert len(response_items) == 2
    streamed_str, function_call = response_items
    assert isinstance(streamed_str, AsyncStreamedStr)
    assert len(await streamed_str.to_string()) > 1  # Check AsyncStreamedStr was cached
    assert isinstance(function_call, FunctionCall)
    assert function_call() is None  # Check FunctionCall is successfully called
