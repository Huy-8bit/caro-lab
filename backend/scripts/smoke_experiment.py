"""Run an isolated distributed training experiment and report game quality.

The experiment launches one authoritative backend plus N stateless self-play
worker processes, exactly like Docker Compose does, but inside a dedicated
work directory. Production ``data/`` and ``checkpoints/`` are never touched.

    python scripts/smoke_experiment.py --work-dir /tmp/caro-smoke \
        --config ../configs/scale.yaml --workers 6 --iterations 20 \
        --set checkpoint_interval=10 --set promotion_games=12
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_games import _print_table, analyze  # noqa: E402


def _parse_value(raw: str):
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parents[1], help="backend directory to run")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--port", type=int, default=8150)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-minutes", type=float, default=60.0)
    parser.add_argument("--bucket", type=int, default=5)
    parser.add_argument("--set", action="append", default=[], help="config override key=value")
    args = parser.parse_args()

    work = args.work_dir.resolve()
    if (work / "data" / "caro.db").exists():
        raise SystemExit(f"{work} already contains an experiment; choose an empty --work-dir")
    (work / "data").mkdir(parents=True, exist_ok=True)
    (work / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text()) or {}
    for item in args.set:
        key, _, value = item.partition("=")
        config[key] = _parse_value(value)
    config["self_play_mode"] = "distributed"
    config["stop_after_promotion"] = False
    (work / "config.yaml").write_text(yaml.safe_dump(config))

    python = sys.executable
    env = {
        **os.environ,
        "CARO_CONFIG": str(work / "config.yaml"),
        "CARO_DATA_DIR": str(work / "data"),
        "CARO_CHECKPOINT_DIR": str(work / "checkpoints"),
        "CARO_WORKER_TOKEN": "smoke-token",
        "PYTHONPATH": str(args.code_dir),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    url = f"http://127.0.0.1:{args.port}"
    logs = (work / "backend.log").open("w")
    server = subprocess.Popen(
        [python, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(args.port)],
        cwd=args.code_dir, env=env, stdout=logs, stderr=subprocess.STDOUT,
    )
    workers: list[subprocess.Popen] = []
    started = time.monotonic()
    try:
        with httpx.Client(base_url=url, timeout=30) as client:
            for _ in range(120):
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            else:
                raise SystemExit("backend did not become healthy")
            response = client.post("/api/training/start", json={"config": config})
            response.raise_for_status()
            worker_log = (work / "workers.log").open("w")
            for index in range(args.workers):
                workers.append(
                    subprocess.Popen(
                        [python, "-m", "app.training.worker"],
                        cwd=args.code_dir,
                        env={
                            **env,
                            "CARO_COORDINATOR_URL": url,
                            "CARO_WORKER_ID": f"smoke-worker-{index + 1}",
                            "CARO_TORCH_THREADS": "1",
                        },
                        stdout=worker_log, stderr=subprocess.STDOUT,
                    )
                )
            last_print = 0.0
            while True:
                status = client.get("/api/status").json()
                elapsed = (time.monotonic() - started) / 60
                if time.monotonic() - last_print > 30:
                    print(
                        f"[{elapsed:5.1f} min] iteration {status['iteration']} phase {status['phase']}"
                        f" games {status['games_completed']} champion v{status['model_version']}",
                        flush=True,
                    )
                    last_print = time.monotonic()
                done = status["iteration"] > args.iterations and status["phase"] != "evaluating"
                if done or status["status"] in {"idle", "error"} or elapsed > args.max_minutes:
                    break
                time.sleep(2)
            client.post("/api/training/stop")
            for _ in range(240):
                if client.get("/api/status").json()["status"] in {"idle", "error"}:
                    break
                time.sleep(0.5)
            (work / "final-status.json").write_text(json.dumps(client.get("/api/status").json(), indent=2))
    finally:
        for process in workers:
            process.send_signal(signal.SIGINT)
        for process in workers:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
    print(f"elapsed {(time.monotonic() - started) / 60:.1f} min")
    report = analyze(work / "data" / "caro.db", args.bucket)
    (work / "report.json").write_text(json.dumps(report, indent=2))
    _print_table(report)


if __name__ == "__main__":
    main()
