"""Utilities for the magentic package."""

import asyncio
import atexit
import threading
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class _AsyncRunner:
    """Manages thread pool and event loops for running async code in sync contexts."""

    def __init__(self, max_workers: int = 2):
        self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._thread_local = threading.local()
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Cleanup the thread pool and event loops on exit."""
        self._thread_pool.shutdown(wait=False, cancel_futures=True)

    def _run_coro_in_thread(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run a coroutine in the thread pool's event loop."""
        if not hasattr(self._thread_local, "loop"):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._thread_local.loop = loop
        else:
            loop = self._thread_local.loop
        return loop.run_until_complete(coro)

    def run_coroutine(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run the coroutine in a separate thread."""
        return self._thread_pool.submit(self._run_coro_in_thread, coro).result()


# Global instance for the package
ASYNC_RUNNER = _AsyncRunner()
