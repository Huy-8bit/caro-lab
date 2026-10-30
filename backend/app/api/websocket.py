from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


router = APIRouter()


async def stream_channel(websocket: WebSocket, channel: str) -> None:
    await websocket.accept()
    broker = websocket.app.state.broker
    queue = broker.subscribe(channel)
    try:
        await websocket.send_json({"type": "connected", "channel": channel})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
                await websocket.send_json(event)
            except TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        pass
    finally:
        broker.unsubscribe(channel, queue)


@router.websocket("/ws/live")
async def live(websocket: WebSocket) -> None:
    await stream_channel(websocket, "live")


@router.websocket("/ws/games/{game_id}")
async def game(websocket: WebSocket, game_id: str) -> None:
    await stream_channel(websocket, f"game:{game_id}")


@router.websocket("/ws/metrics")
async def metrics(websocket: WebSocket) -> None:
    await stream_channel(websocket, "metrics")


@router.websocket("/ws/evaluations")
async def evaluations(websocket: WebSocket) -> None:
    await stream_channel(websocket, "evaluation")
