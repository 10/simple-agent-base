from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import suppress
from typing import TypeVar

T = TypeVar("T")
_SENTINEL = object()


def ensure_sync_allowed(api_name: str, async_hint: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return

    raise RuntimeError(
        f"{api_name} cannot be used inside a running event loop. Use '{async_hint}' instead."
    )


class SyncRuntime:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._closed = False

    def run(self, awaitable_factory: Callable[[], Awaitable[T]]) -> T:
        loop = self._ensure_started()
        future = asyncio.run_coroutine_threadsafe(awaitable_factory(), loop)
        return future.result()

    def iterate(self, async_iterable_factory: Callable[[], AsyncIterator[T]]) -> Iterator[T]:
        loop = self._ensure_started()
        output_queue: queue.Queue[T | BaseException | object] = queue.Queue()
        stop_requested = threading.Event()

        async def consume() -> None:
            async_iterable = async_iterable_factory()
            try:
                async for item in async_iterable:
                    output_queue.put(item)
                    if stop_requested.is_set():
                        break
            except BaseException as exc:
                output_queue.put(exc)
            finally:
                aclose = getattr(async_iterable, "aclose", None)
                if callable(aclose):
                    with suppress(Exception):
                        await aclose()
                output_queue.put(_SENTINEL)

        future = asyncio.run_coroutine_threadsafe(consume(), loop)

        try:
            while True:
                item = output_queue.get()
                if item is _SENTINEL:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop_requested.set()
            if not future.done():
                future.cancel()
            with suppress(concurrent.futures.CancelledError, TimeoutError):
                future.result(timeout=1.0)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread

        if loop is None or thread is None:
            return

        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._closed:
                raise RuntimeError("Sync runtime has already been closed.")

            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run_loop,
                    name="simple-agent-base-sync-runtime",
                    daemon=True,
                )
                self._thread.start()

        self._ready.wait()
        if self._loop is None:
            raise RuntimeError("Sync runtime failed to start.")
        return self._loop

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()

        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()
