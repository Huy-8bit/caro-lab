# Caro Lab

Caro Lab is a complete, local-first platform for training and observing a chess agent through genuine self-play reinforcement learning. It combines a small AlphaZero-inspired PyTorch engine with a FastAPI service and a polished React dashboard for live games, replay, analytics, checkpoint management, and model battles.

This is an educational, personal-computer-scale implementation. It demonstrates the full learning loop; it is not a reproduction of AlphaZero's compute budget or playing strength, and it does not claim an absolute Elo rating.

## What is real

- Every move comes from `python-chess` legal move generation and neural-network-guided MCTS.
- Training targets are MCTS visit distributions and final outcomes from the correct side-to-move perspective.
- Policy, value, and combined losses come from actual PyTorch optimizer steps.
- The dashboard reads metrics, games, checkpoints, logs, and evaluation results from the backend. Offline state is shown honestly.
- Completed games, PGN, per-ply FEN, metrics, runtime state, checkpoints, and evaluation results survive restarts.
- Live boards receive sampled moves through subscription-based WebSockets; training does not depend on browser connections.

## Architecture

```text
Self-play workers ──> MCTS ──> legal chess games ──> replay buffer
       │                 │                              │
       │                 └── policy/value network <────┘ optimizer
       │                                      │
       ├── WebSocket broker ──> React live arena
       └── SQLite + PGN/FEN ──> history / analytics
                                              │
PyTorch checkpoints <── evaluation arena <────┘
```

```text
backend/app/
├── api/          Typed REST and WebSocket routes
├── chess/        Board encoding, move indexing, MCTS
├── training/     Network, replay buffer, self-play, optimizer, checkpoints
├── evaluation/   Random, heuristic, and checkpoint opponents
├── storage/      SQLite repository and retention
└── core/         Thread-safe event fan-out

app/              React + TypeScript dashboard
configs/          Quick and normal training profiles
data/             SQLite runtime data (ignored except .gitkeep)
checkpoints/      PyTorch checkpoint files (ignored except .gitkeep)
```

The dashboard uses React 19, TypeScript, Tailwind's build pipeline, accessible CSS-native charts, Lucide icons, and a Vite-powered Vinext runtime. The training API remains the sole authority for chess state and validation.

## Learning system

### Board representation

The network receives 20 planes of shape `8×8`: 12 piece planes, side to move, four castling-right planes, en-passant target, halfmove clock, and fullmove number. Moves use a deterministic `from × to × promotion` index with 20,480 possible policy outputs. Legal move masking is applied before softmax.

### Policy-value network

A configurable convolutional residual tower feeds:

- a policy head trained against normalized MCTS visit counts;
- a `tanh` value head trained against the final result from the player-to-move perspective.

The optimizer minimizes cross-entropy policy loss plus mean-squared value loss. Gradient clipping and AdamW are included. Device selection prefers CUDA, then Apple Metal (MPS), then CPU; a device can also be forced explicitly.

### MCTS

Each search expands legal moves using policy priors, selects with PUCT, evaluates leaves with the value head, flips perspective on every backpropagation edge, and produces a visit-count policy. Root Dirichlet noise and early-move temperature provide self-play exploration. `python-chess` supplies castling, promotion, en passant, checkmate, stalemate, repetition claims, and move-count draw rules.

### Training loop

Each iteration runs the configured number of self-play games across bounded worker threads, appends samples to a capacity-limited replay buffer, performs real optimizer updates, and records metrics. Training progress is atomically overwritten in one `training-latest.pt` candidate file. After a configurable multi-iteration cycle, the candidate plays a color-balanced, opening-diverse promotion match against the current champion. Only a candidate that reaches the required score becomes a numbered checkpoint; a rejected candidate is rolled back to the champion. By default, training stops cleanly after a new numbered model is saved; this can be disabled in Configuration for continuous training. Pause takes effect at move and optimizer boundaries, and Stop preserves the rolling candidate state.

## Quick start with Docker

Requirements: Docker Desktop with Compose.

```bash
docker compose up --build
```

Open `http://localhost:3000`. The API and interactive documentation are at `http://localhost:8000` and `http://localhost:8000/docs`.

The default Compose profile is intentionally small. Data is mounted into `./data` and checkpoints into `./checkpoints`.

For horizontally scaled self-play on one Docker host, run one authoritative trainer and four stateless actor containers:

```bash
docker compose -f docker-compose.yml -f docker-compose.scale.yml up --build --scale selfplay-worker=4
```

The trainer is the only process that owns the optimizer, replay buffer, SQLite database, and checkpoints. Workers lease versioned jobs over authenticated internal HTTP, cache candidate snapshots, stream sampled live moves, and return compressed trajectories. Increasing `--scale selfplay-worker=N` changes self-play capacity without creating competing model writers.

## Local development

Requirements: Node.js 22+, Python 3.11–3.13, and `uv` (recommended).

Install the backend:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r backend/requirements.txt
```

Start the API:

```bash
cd backend
CARO_CONFIG=../configs/quick.yaml ../.venv/bin/python -m uvicorn app.main:app --reload
```

In another terminal, install and start the dashboard:

```bash
npm install
npm run dev
```

The dashboard defaults to `http://localhost:8000`. Override it with `VITE_API_URL` when needed.

## Training profiles

- `configs/quick.yaml`: compact network with 32 MCTS simulations, four games per iteration, and a 10-iteration promotion cycle; suitable for meaningful CPU-scale experiments.
- `configs/normal.yaml`: larger network, two workers, 96 simulations, larger replay and batch sizes, and longer promotion matches; intended for long local runs.
- Custom: copy either file and pass it via `CARO_CONFIG`, or edit all important settings on the Configuration page.

Important parameters include games and workers per iteration, MCTS simulations, batch size, learning rate, replay capacity, optimizer updates, candidate-cycle length, promotion games and score threshold, evaluation search depth, opening diversity, temperature, Dirichlet noise, PUCT, architecture size, retention, event sampling, device, and seed.

### CPU only

Set `device: cpu` in the YAML file or choose `cpu` in a custom API configuration.

### CUDA or Apple Silicon

`device: auto` selects CUDA when available, otherwise MPS on Apple Silicon, otherwise CPU. CUDA hosts must install the appropriate PyTorch wheel for their driver before starting the backend. MPS ships with the macOS PyTorch wheel.

## Dashboard guide

- **Overview** — authoritative status, throughput, loss, win/draw split, resource usage, recent games, and a live game preview.
- **Live Arena** — subscribe to an active worker, pause only visual playback, flip the board, and inspect move-level visits and values.
- **Game History** — persistent records with timeline scrubbing, first/previous/next/last controls, autoplay speed, FEN inspection, board orientation, and PGN download.
- **Training Analytics** — actual optimization losses and throughput plus color-balanced evaluation results.
- **Checkpoints** — inspect, download, or resume promoted champion models; rolling candidates do not clutter this registry.
- **Model Battle** — play a candidate against random, a capture-first heuristic, or another checkpoint with alternating colors.
- **Configuration** — launch a new run with custom self-play, MCTS, replay, optimizer, and network settings.
- **System Logs** — bounded structured worker and checkpoint events.

## API highlights

```text
GET  /health
GET  /api/status
POST /api/training/start|pause|resume|stop
GET  /api/games
GET  /api/games/{id}
GET  /api/games/{id}/pgn
GET  /api/metrics
GET  /api/checkpoints
GET  /api/checkpoints/{version}/download
POST /api/evaluations
WS   /ws/live
WS   /ws/games/{id}
WS   /ws/metrics
WS   /ws/evaluations
```

Live move delivery is channel-based with bounded per-client queues. A slow or disconnected viewer cannot stall workers or cause unbounded memory growth.

## Tests and validation

```bash
cd backend
../.venv/bin/python -m pytest

cd ..
npm run lint
npx tsc --noEmit
npm run build
```

The suite covers chess special rules, checkmate and draw outcomes, encoding, move indexing, policy/value shapes, legal MCTS output, perspective handling, replay capacity, optimizer weight changes, checkpoint round-trips, SQLite game persistence, PGN replay, HTTP/WebSocket contracts, and a complete self-play → optimize → persist → checkpoint integration pipeline.

For a visible smoke experiment, start the API with the quick profile, start training from the dashboard, wait for a checkpoint, stop gracefully, restart the API, and verify that the same game count, iteration, and model version are restored.

## Persistence and recovery

SQLite uses WAL mode and indexed game/metric queries. Model tensors stay outside the database as `.pt` files. On startup, Caro Lab reloads the latest valid checkpoint and merges it with the persisted runtime counters. Completed games and evaluation records are never reconstructed from memory. Storage retention is configurable.

## Known limitations

- The compact move encoding is deliberately simpler and larger than AlphaZero's directional action encoding.
- The replay buffer is bounded but in memory; durable games are retained, while samples are regenerated after a process restart.
- Multi-worker inference shares one model in a single service process. This is appropriate for a workstation, not a distributed GPU cluster.
- Evaluation is relative and color-balanced but intentionally has no absolute Elo claim.
- Training at workstation scale remains compute-intensive. Candidates use material bootstrapping only when a self-play game reaches its safety limit, while true checkmates and rule-based draws remain authoritative.
- Authentication and multi-user experiment ownership are outside the local-first scope.

## License

MIT. Contributions are welcome, especially improvements to distributed self-play, opening diversity, richer checkpoint gating, and calibrated external-engine evaluation.
