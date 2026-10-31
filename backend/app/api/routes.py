from __future__ import annotations

import copy
import hmac
import os
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse

from app.api.schemas import (
    EvaluationRequest,
    StartTrainingRequest,
    WorkerClaimRequest,
    WorkerEventRequest,
    WorkerResultRequest,
)
from app.config import TrainingConfig
from app.evaluation.arena import evaluate_checkpoint


router = APIRouter(prefix="/api")


def services(request: Request):
    return request.app.state.engine, request.app.state.repository, request.app.state.broker


def authorize_worker(request: Request) -> None:
    expected = os.getenv("CARO_WORKER_TOKEN", "local-scale-token")
    supplied = request.headers.get("X-Caro-Worker-Token", "")
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid worker token")


@router.get("/status")
def status(request: Request):
    engine, _, _ = services(request)
    return engine.snapshot()


@router.post("/training/start")
def start_training(payload: StartTrainingRequest, request: Request):
    engine, _, _ = services(request)
    config = TrainingConfig.from_dict(payload.config.model_dump()) if payload.config else None
    try:
        return engine.start(config, payload.resume_version)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/training/pause")
def pause_training(request: Request):
    engine, _, _ = services(request)
    try:
        return engine.pause()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/training/resume")
def resume_training(request: Request):
    engine, _, _ = services(request)
    try:
        return engine.resume()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/training/stop")
def stop_training(request: Request):
    engine, _, _ = services(request)
    return engine.stop()


@router.post("/workers/claim")
def worker_claim(payload: WorkerClaimRequest, request: Request):
    authorize_worker(request)
    engine, _, _ = services(request)
    job = engine.claim_worker_job(payload.worker_id)
    return job if job is not None else Response(status_code=204)


@router.get("/workers/model")
def worker_model(revision: int, request: Request):
    authorize_worker(request)
    engine, _, _ = services(request)
    try:
        path = engine.worker_model_path(revision)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return FileResponse(path)


@router.post("/workers/results")
def worker_result(payload: WorkerResultRequest, request: Request):
    authorize_worker(request)
    engine, _, _ = services(request)
    try:
        accepted = engine.submit_worker_result(
            payload.worker_id,
            payload.job_id,
            payload.revision,
            payload.record,
            payload.samples_blob,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not accepted:
        raise HTTPException(status_code=409, detail="Job is stale or already completed")
    return {"accepted": True}


@router.post("/workers/events")
def worker_event(payload: WorkerEventRequest, request: Request):
    authorize_worker(request)
    engine, _, _ = services(request)
    accepted = engine.ingest_worker_event(
        payload.worker_id,
        payload.job_id,
        payload.revision,
        payload.event,
    )
    return {"accepted": accepted}


@router.get("/games")
def games(request: Request, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    _, repository, _ = services(request)
    return {"items": repository.list_games(limit, offset)}


@router.get("/games/{game_id}")
def game(game_id: str, request: Request):
    _, repository, _ = services(request)
    record = repository.get_game(game_id)
    if not record:
        raise HTTPException(status_code=404, detail="Game not found")
    return record


@router.get("/games/{game_id}/pgn", response_class=PlainTextResponse)
def game_pgn(game_id: str, request: Request):
    _, repository, _ = services(request)
    record = repository.get_game(game_id)
    if not record:
        raise HTTPException(status_code=404, detail="Game not found")
    return PlainTextResponse(
        record["pgn"],
        headers={"Content-Disposition": f'attachment; filename="{game_id}.pgn"'},
    )


@router.get("/metrics")
def metrics(request: Request, limit: int = Query(200, ge=1, le=5_000)):
    _, repository, _ = services(request)
    return {"items": repository.list_metrics(limit)}


@router.get("/checkpoints")
def checkpoints(request: Request):
    _, repository, _ = services(request)
    return {"items": repository.list_checkpoints()}


@router.get("/checkpoints/{version}/download")
def checkpoint_download(version: int, request: Request):
    _, repository, _ = services(request)
    metadata = repository.get_checkpoint(version)
    if not metadata or not Path(metadata["path"]).exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return FileResponse(metadata["path"], filename=Path(metadata["path"]).name)


@router.get("/evaluations")
def evaluations(request: Request):
    _, repository, _ = services(request)
    return {"items": repository.list_evaluations()}


@router.post("/evaluations", status_code=202)
def evaluation(payload: EvaluationRequest, request: Request):
    engine, repository, broker = services(request)
    model_version = payload.model_version if payload.model_version is not None else engine.model_version
    checkpoint_path = None
    if payload.model_version is not None and payload.model_version != engine.model_version:
        selected = repository.get_checkpoint(payload.model_version)
        if not selected:
            raise HTTPException(status_code=404, detail="Candidate checkpoint not found")
        candidate, _ = __import__("app.training.checkpoints", fromlist=["load_checkpoint"]).load_checkpoint(
            selected["path"], engine.device
        )
    else:
        candidate = copy.deepcopy(engine.model).to(engine.device)
    if payload.opponent == "checkpoint":
        if payload.opponent_version is None:
            raise HTTPException(status_code=422, detail="opponent_version is required")
        opponent_metadata = repository.get_checkpoint(payload.opponent_version)
        if not opponent_metadata:
            raise HTTPException(status_code=404, detail="Opponent checkpoint not found")
        checkpoint_path = opponent_metadata["path"]

    def run() -> None:
        evaluate_checkpoint(
            candidate,
            model_version,
            engine.device,
            repository,
            payload.opponent,
            payload.games,
            payload.simulations,
            checkpoint_path=checkpoint_path,
            event_callback=lambda event: broker.publish("evaluation", event),
            seed=engine.config.random_seed,
        )

    threading.Thread(target=run, name="caro-evaluation", daemon=True).start()
    return {"status": "started", "model_version": model_version, "opponent": payload.opponent}


@router.get("/logs")
def logs(request: Request):
    engine, _, _ = services(request)
    return {"items": list(engine.logs)}
