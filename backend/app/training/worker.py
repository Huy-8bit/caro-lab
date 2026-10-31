from __future__ import annotations

import os
import socket
import time
import traceback
from typing import Any

import httpx
import torch

from app.chess.mcts import MCTS
from app.training.distributed import encode_samples, load_worker_model
from app.training.network import choose_device
from app.training.self_play import play_self_play_game


COORDINATOR_URL = os.getenv("CARO_COORDINATOR_URL", "http://backend:8000")
WORKER_TOKEN = os.getenv("CARO_WORKER_TOKEN", "local-scale-token")
WORKER_ID = os.getenv("CARO_WORKER_ID", socket.gethostname())
WORKER_DEVICE = os.getenv("CARO_WORKER_DEVICE", "cpu")
POLL_SECONDS = float(os.getenv("CARO_WORKER_POLL_SECONDS", "0.5"))


def run() -> None:
    torch.set_num_threads(max(1, int(os.getenv("CARO_TORCH_THREADS", "1"))))
    device = choose_device(WORKER_DEVICE)
    headers = {"X-Caro-Worker-Token": WORKER_TOKEN}
    model = None
    revision = -1
    with httpx.Client(base_url=COORDINATOR_URL, headers=headers, timeout=30.0) as client:
        print(f"worker={WORKER_ID} coordinator={COORDINATOR_URL} device={device}", flush=True)
        while True:
            try:
                response = client.post("/api/workers/claim", json={"worker_id": WORKER_ID})
                if response.status_code == 204:
                    time.sleep(POLL_SECONDS)
                    continue
                response.raise_for_status()
                job = response.json()
                if model is None or revision != int(job["revision"]):
                    weights = client.get("/api/workers/model", params={"revision": job["revision"]})
                    weights.raise_for_status()
                    model, metadata = load_worker_model(weights.content, device)
                    revision = int(metadata["revision"])
                    print(f"worker={WORKER_ID} loaded revision={revision}", flush=True)
                config = job["config"]
                search = MCTS(
                    model,
                    device,
                    simulations=int(config["mcts_simulations"]),
                    c_puct=float(config["c_puct"]),
                    dirichlet_alpha=float(config["dirichlet_alpha"]),
                    dirichlet_epsilon=float(config["dirichlet_epsilon"]),
                    repetition_penalty=float(config["repetition_penalty"]),
                    seed=int(job["seed"]),
                )

                def report(event: dict[str, Any]) -> None:
                    if event.get("type") != "move":
                        return
                    sample_rate = max(1, int(config["event_sample_rate"]))
                    if int(event["ply"]) % sample_rate:
                        return
                    try:
                        client.post(
                            "/api/workers/events",
                            json={
                                "worker_id": WORKER_ID,
                                "job_id": job["job_id"],
                                "revision": revision,
                                "event": event,
                            },
                            timeout=3.0,
                        )
                    except httpx.HTTPError:
                        pass

                result = play_self_play_game(
                    search=search,
                    iteration=int(job["iteration"]),
                    model_version=int(job["model_version"]),
                    max_moves=int(config["max_game_moves"]),
                    temperature_moves=int(config["temperature_moves"]),
                    temperature=float(config["temperature"]),
                    draw_penalty=float(config["draw_penalty"]),
                    event_callback=report,
                )
                submission = {
                    "worker_id": WORKER_ID,
                    "job_id": job["job_id"],
                    "revision": revision,
                    "record": result.record,
                    "samples_blob": encode_samples(result.samples),
                }
                while True:
                    completed = client.post("/api/workers/results", json=submission, timeout=300.0)
                    if completed.status_code in {200, 409}:
                        break
                    completed.raise_for_status()
                print(
                    f"worker={WORKER_ID} completed job={job['job_id']} result={result.record['result']}",
                    flush=True,
                )
            except KeyboardInterrupt:
                return
            except Exception as exc:
                print(f"worker={WORKER_ID} error={type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                time.sleep(2)


if __name__ == "__main__":
    run()
