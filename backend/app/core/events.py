from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any


class EventBroker:
    """Thread-safe fan-out broker with per-game subscriptions."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def subscribe(self, channel: str, maxsize: int = 128) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._subscribers[channel].add(queue)
        return queue

    def unsubscribe(self, channel: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers[channel].discard(queue)

    def publish(self, channel: str, event: dict[str, Any]) -> None:
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self._deliver, channel, event)

    def _deliver(self, channel: str, event: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers.get(channel, ())):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)
