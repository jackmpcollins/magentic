# Asyncio

Asynchronous functions / coroutines can be used to concurrently query the LLM. This can greatly increase the overall speed of generation, and also allow other asynchronous code to run while waiting on LLM output. In the below example, the LLM generates a description for each US president while it is waiting on the next one in the list. Measuring the characters generated per second shows that this example achieves a 7x speedup over serial processing.

```python
import asyncio
from time import time
from typing import AsyncIterable

from magentic import prompt


@prompt("List ten presidents of the United States")
async def iter_presidents() -> AsyncIterable[str]:
    ...


@prompt("Tell me more about {topic}")
async def tell_me_more_about(topic: str) -> str:
    ...


# For each president listed, generate a description concurrently
start_time = time()
tasks = []
async for president in await iter_presidents():
    # Use asyncio.create_task to schedule the coroutine for execution before awaiting it
    # This way descriptions will start being generated while the list of presidents is still being generated
    task = asyncio.create_task(tell_me_more_about(president))
    tasks.append(task)

descriptions = await asyncio.gather(*tasks)

# Measure the characters per second
total_chars = sum(len(desc) for desc in descriptions)
time_elapsed = time() - start_time
print(total_chars, time_elapsed, total_chars / time_elapsed)
# 24575 28.70 856.07


# Measure the characters per second to describe a single president
start_time = time()
out = await tell_me_more_about("George Washington")
time_elapsed = time() - start_time
print(len(out), time_elapsed, len(out) / time_elapsed)
# 2206 18.72 117.78
```
