from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router as api_router
from app.api.websocket import router as websocket_router
from app.config import ROOT, TrainingConfig
from app.core.events import EventBroker
from app.storage.repository import Repository
from app.training.engine import TrainingEngine


DATA_DIR = Path(os.getenv("CARO_DATA_DIR", ROOT / "data"))
CHECKPOINT_DIR = Path(os.getenv("CARO_CHECKPOINT_DIR", ROOT / "checkpoints"))
CONFIG_PATH = os.getenv("CARO_CONFIG")
if os.getenv("CARO_TORCH_THREADS"):
    torch.set_num_threads(max(1, int(os.environ["CARO_TORCH_THREADS"])))
repository = Repository(DATA_DIR / "caro.db")
broker = EventBroker()
config = TrainingConfig.from_yaml(CONFIG_PATH) if CONFIG_PATH else TrainingConfig()
engine = TrainingEngine(repository, broker, CHECKPOINT_DIR, config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    broker.bind(asyncio.get_running_loop())
    yield
    if engine.status in {"running", "paused", "stopping"}:
        engine.stop(wait=True)


app = FastAPI(
    title="Caro Lab API",
    description="Authoritative self-play chess training and visualization API",
    version=__version__,
    lifespan=lifespan,
)
app.state.repository = repository
app.state.broker = broker
app.state.engine = engine
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CARO_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
app.include_router(websocket_router)


@app.get("/health")
def health() -> dict[str, str | int]:
    return {
        "status": "ok",
        "version": __version__,
        "training": engine.status,
        "phase": engine.phase,
        "champion": engine.model_version,
        "workers": len(engine._active_workers()),
    }
