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

- a policy head trained against the search's improved policy (completed-Q by default, or normalized visit counts);
- a `tanh` value head trained against the final result from the player-to-move perspective.

The optimizer minimizes cross-entropy policy loss plus mean-squared value loss. Gradient clipping and AdamW are included. Device selection prefers CUDA, then Apple Metal (MPS), then CPU; a device can also be forced explicitly.

### MCTS

Each search expands legal moves using policy priors, selects with PUCT, evaluates leaves with the value head, flips perspective on every backpropagation edge, and produces a visit-count policy. Root Dirichlet noise and early-move temperature provide self-play exploration; evaluation matches use neither.

The search runs on the real game history (back to the last capture or pawn move), so it sees repetitions that involve positions played before the root. A leaf is scored exactly, without a network call, when:

- the game is over under the shared rules in `app/chess/rules.py` (checkmate, stalemate, insufficient material, fifty-move rule at 100 half-moves, threefold repetition once a position has actually occurred three times, fivefold/75-move);
- the simulated line returns to a position already on its own path — a cycle the side to move could repeat forever — which is scored as a draw.

**Policy targets (experimental option).** Training uses AlphaZero visit-count targets by default. At 32 simulations PUCT visits mostly echo the prior: in a smoke run that already had history-aware search and contempt, the search had proven a repeating move worse than a visited alternative in 35 of 40 cases, yet still played it in 29 because it had the most visits. `policy_target: completed_q` switches targets and move choice to the completed-Q improved policy from Gumbel AlphaZero/MuZero (Danihelka et al., 2022). In our 30-iteration smoke run it made repetition *worse* (66% vs 48% of games in iterations 26–30), because it follows the network's own Q faster, so it stays opt-in. The structural fix for prior dominance is a larger search budget.

Draws inside the search carry a small `draw_contempt` from the point of view of the player choosing the move (−c for that player, +c for the opponent, so every edge stays zero-sum). Value **targets** for drawn games remain exactly 0.

### Training loop

Each iteration runs the configured number of self-play games (local threads or distributed workers), appends samples to a capacity-limited replay buffer, performs real optimizer updates, and records losses together with separate self-play and optimizer timings.

Replay sampling is outcome-aware. Each game's samples carry a weight from its outcome class (decisive 1.0, rule draws `draw_sample_weight`, repetition draws `repetition_sample_weight`, move-limit adjudications `adjudicated_sample_weight`), long games are normalised so a 300-ply shuffle does not outweigh a short mating game by sample count, positions repeated within a game are recorded once, and non-decisive samples are capped at `replay_max_draw_fraction` of the sampling mass whenever decisive samples exist. Each self-play game starts from a few randomly sampled opening plies (`selfplay_opening_plies`) that are not used as training targets.

After `checkpoint_interval` iterations the candidate plays the champion (see *Promotion gate*). Training progress is atomically overwritten in one `training-latest.pt` candidate file; only promoted models get numbered checkpoints, and every promoted checkpoint is kept. By default training stops right after a model is promoted.

### Promotion gate

- The very first model (v1) is registered as a **bootstrap** checkpoint: it has no opponent, never played a match and is shown as "Bootstrap · no match played".
- Every later candidate plays `promotion_games` (default 40, rounded up to an even number) against the current champion: game pairs share a randomized opening with colours swapped, both sides search `evaluation_simulations` without Dirichlet noise and play greedily. In distributed mode these games run on the self-play workers.
- The candidate is promoted only if its score is at least `promotion_threshold` (default 55%) **and** the match passes the quality gate: repetition-ended games at most `gate_max_repetition_rate` (50%) and decisive games at least `gate_min_decisive_rate` (10%).
- A rejected candidate leaves the champion untouched; the reasons are stored with the evaluation and shown in the dashboard. The trainer keeps its weights and continues learning, unless the candidate scored below `reject_reset_score` (35%), in which case it restarts from the champion weights.
- Live match state (games done/total, provisional W/D/L, games in progress, elapsed time, ETA, termination breakdown, decision) is persisted in SQLite and served by `GET /api/evaluations/progress`, `GET /api/status` (`evaluation`) and `WS /ws/evaluations` (which sends the current state on connect).

### Why loss alone is misleading

Loss measures how well the network fits its own recent self-play targets. In a 30-iteration reproduction of the draw collapse with the previous pipeline, self-play went from 24% to 92% draws and from 76% to 8% checkmates while total loss fell from 7.7 to 3.6, because draw targets are trivially easy to fit. The Training Analytics page therefore shows rolling draw, repetition, checkmate and decisive rates per iteration next to the losses, and warns when draws or repetitions keep rising while loss falls. `scripts/analyze_games.py` gives the same breakdown from any database (read-only).

## Quick start with Docker

Requirements: Docker Desktop with Compose.

```bash
docker compose up --build
```

Open `http://localhost:3000`. The API and interactive documentation are at `http://localhost:8000` and `http://localhost:8000/docs`.

The default Compose profile is intentionally small. Data is mounted into `./data` and checkpoints into `./checkpoints`.

For horizontally scaled self-play on one Docker host, run one authoritative trainer and six stateless actor containers (the default, sized for an 8-core machine):

```bash
docker compose -f docker-compose.yml -f docker-compose.scale.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.scale.yml ps        # backend, dashboard and 6 workers healthy
curl -X POST http://localhost:8000/api/training/start -H 'Content-Type: application/json' -d '{}'
```

Change capacity at any time with `--scale selfplay-worker=N`; the iteration in progress is not lost. The trainer is the only process that owns the optimizer, replay buffer, SQLite database and checkpoints (only the backend mounts `./data` and `./checkpoints`). Workers lease one job at a time, heartbeat while playing, and return compressed trajectories or evaluation results. A lease without a heartbeat for `worker_job_timeout` seconds is requeued; a stopped worker releases its job immediately; a restarted worker's abandoned job is requeued when it claims again; only the first result for a job is accepted.

Self-play is one continuous job stream rather than a batch per iteration: the backend keeps `games_per_iteration` plus one look-ahead job per active worker outstanding, each iteration takes the first `games_per_iteration` results, and games still running at the boundary count towards the next iteration (recorded with the iteration and model they were played with). Queued jobs are switched to the newest model before each iteration, and promotion matches cancel queued self-play so evaluation games start first. This keeps workers busy instead of waiting for the longest game of every iteration. MCTS evaluates only the policy logits of legal moves (`PolicyValueNetwork.evaluate_moves`), which is numerically the same as the full 20,480-way head but several times faster on CPU.

`endgame_curriculum_fraction` (experimental, off by default) starts that share of self-play games from random legal reduced-material positions (`app/chess/endgames.py`); only the start square layout is random, games are played and scored under the normal rules, and promotion matches always use the standard start. Curriculum games are excluded from the draw/repetition/checkmate rates and reported separately. `scripts/probe_endgames.py` measures whether a checkpoint values won endgames positively and can mate KQ v K / KR v K. In a 30-iteration smoke run (64 simulations, 6 workers) the curriculum did not improve either measure (KQ v K value ≈ 0, mate rate 0–5% with and without it), so it stays off until a longer run shows a benefit.

Compare promoted models with the same protocol as the gate, either from the dashboard (Model Battle → Previous checkpoint) or from the command line (read-only):

```bash
docker compose exec backend python scripts/compare_models.py /checkpoints/model-v0002.pt /checkpoints/model-v0001.pt --games 40 --simulations 48 --parallel 2
docker compose exec backend python scripts/analyze_games.py /data/caro.db --bucket 10 --last 500
```

To verify a change from scratch without touching production data, use an isolated directory:

```bash
cd backend && ../.venv/bin/python scripts/smoke_experiment.py --work-dir /tmp/caro-smoke --config ../configs/scale.yaml --workers 6 --iterations 30
```

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
- **Training Analytics** — per-iteration policy/value/total loss (labelled small multiples), rolling draw/repetition/checkmate/decisive rates with quality warnings, self-play-only vs. including-optimizer throughput, and head-to-head match results with Elo estimates. Training loss and playing strength are shown as separate signals.
- **Checkpoints** — inspect, download, or resume promoted champion models; the bootstrap model is labelled as such; rolling candidates do not clutter this registry.
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
GET  /api/analytics/iterations
GET  /api/checkpoints
GET  /api/checkpoints/{version}/download
GET  /api/evaluations
GET  /api/evaluations/progress
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

SQLite uses WAL mode and indexed game/metric queries. Before a schema upgrade the backend writes a timestamped copy next to the database (`caro.pre-migration-<UTC>.db`) and only adds columns; existing rows are never rewritten. Starting with a network size that differs from the current champion is refused instead of silently resetting the run — use a separate data/checkpoint directory for a new experiment. Model tensors stay outside the database as `.pt` files. On startup, Caro Lab reloads the latest valid checkpoint and merges it with the persisted runtime counters. Completed games and evaluation records are never reconstructed from memory. Storage retention is configurable.

## Known limitations

- The compact move encoding is deliberately simpler and larger than AlphaZero's directional action encoding.
- The replay buffer is bounded but in memory; durable games are retained, while samples are regenerated after a process restart.
- Multi-worker inference shares one model in a single service process. This is appropriate for a workstation, not a distributed GPU cluster.
- Evaluation is relative and color-balanced but intentionally has no absolute Elo claim.
- Training at workstation scale remains compute-intensive. Candidates use material bootstrapping only when a self-play game reaches its safety limit, while true checkmates and rule-based draws remain authoritative.
- Authentication and multi-user experiment ownership are outside the local-first scope.

## License

MIT. Contributions are welcome, especially improvements to distributed self-play, opening diversity, richer checkpoint gating, and calibrated external-engine evaluation.
