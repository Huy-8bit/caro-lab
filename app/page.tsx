"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity, Archive, BarChart3, BrainCircuit, Check, ChevronFirst, ChevronLast,
  ChevronLeft, ChevronRight, CirclePause, CirclePlay, Clock3, Cpu, Database,
  Download, FlaskConical, Gauge, History, LayoutDashboard, ListTree, LoaderCircle,
  Menu, MonitorDot, Pause, Play, RotateCcw, Settings2, ShieldCheck, Swords,
  TerminalSquare, Trophy, WifiOff, X,
} from "lucide-react";
import { ChessBoard } from "./components/ChessBoard";
import { ThroughputChart, TrainingLossChart } from "./components/TrainingCharts";
import type { ActiveGame, Checkpoint, Evaluation, Game, Metric, TrainingConfig, TrainingStatus } from "./types";

const API = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const DEFAULT_CONFIG: TrainingConfig = {
  experiment_name: "quick-start", games_per_iteration: 4, parallel_workers: 2,
  self_play_mode: "local", worker_sync_interval: 5, worker_job_timeout: 300,
  mcts_simulations: 32, max_game_moves: 120, batch_size: 32, learning_rate: 0.0005,
  weight_decay: 0.0001, replay_capacity: 20000, training_updates: 16,
  checkpoint_interval: 10, evaluation_frequency: 10, promotion_games: 8,
  promotion_threshold: 0.55, evaluation_simulations: 32,
  evaluation_opening_plies: 4, stop_after_promotion: true, autosave_interval: 2, temperature_moves: 20,
  temperature: 1, dirichlet_alpha: 0.3, dirichlet_epsilon: 0.25, c_puct: 1.5,
  repetition_penalty: 0.01, draw_penalty: 0.05,
  random_seed: 7, device: "auto", channels: 24, residual_blocks: 1,
  history_retention: 5000, event_sample_rate: 1,
};

const EMPTY_STATUS: TrainingStatus = {
  status: "idle", phase: "idle", iteration: 0, step: 0, model_version: 0,
  candidate_progress: 0, candidate_target: 10, last_promotion: null, games_completed: 0,
  worker_count: 0, workers: [],
  active_games: [], runtime_seconds: 0, games_per_minute: 0, average_game_length: 0,
  white_win_rate: 0, black_win_rate: 0, draw_rate: 0, policy_loss: null,
  value_loss: null, total_loss: null, replay_size: 0, device: "—",
  system: { cpu_percent: 0, ram_percent: 0, process_mb: 0, platform: "—" },
  last_error: null, config: DEFAULT_CONFIG,
};

const nav = [
  ["overview", "Overview", LayoutDashboard], ["arena", "Live Arena", MonitorDot],
  ["history", "Game History", History], ["analytics", "Training Analytics", BarChart3],
  ["models", "Checkpoints", Archive], ["battle", "Model Battle", Swords],
  ["config", "Configuration", Settings2], ["logs", "System Logs", TerminalSquare],
] as const;

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText })) as { detail?: string };
    throw new Error(body.detail ?? "Request failed");
  }
  return response.json();
}

function duration(seconds: number) {
  const hours = Math.floor(seconds / 3600).toString().padStart(2, "0");
  const minutes = Math.floor((seconds % 3600) / 60).toString().padStart(2, "0");
  const secs = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${hours}:${minutes}:${secs}`;
}

function percent(value: number) { return `${(value * 100).toFixed(1)}%`; }
function number(value: number | null, digits = 3) { return value == null ? "—" : value.toFixed(digits); }
function bytes(value: number) { return value < 1_000_000 ? `${(value / 1000).toFixed(0)} KB` : `${(value / 1_000_000).toFixed(1)} MB`; }

function StatusPill({ connected, status }: { connected: boolean; status: TrainingStatus["status"] }) {
  return <span className={`statusPill ${connected ? status : "offline"}`}><i />{connected ? status : "offline"}</span>;
}

function Empty({ icon: Icon, title, copy }: { icon: typeof Activity; title: string; copy: string }) {
  return <div className="empty"><Icon size={24} /><strong>{title}</strong><p>{copy}</p></div>;
}

function MetricCard({ label, value, note, icon: Icon }: { label: string; value: string; note: string; icon: typeof Activity }) {
  return <article className="metricCard"><div><span>{label}</span><Icon size={15} /></div><strong>{value}</strong><small>{note}</small></article>;
}

export default function Home() {
  const [view, setView] = useState<(typeof nav)[number][0]>("overview");
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<TrainingStatus>(EMPTY_STATUS);
  const [games, setGames] = useState<Game[]>([]);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [evaluations, setEvaluations] = useState<Evaluation[]>([]);
  const [logs, setLogs] = useState<{ level: string; message: string; time: string }[]>([]);
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [selectedGame, setSelectedGame] = useState<Game | null>(null);
  const [selectedActive, setSelectedActive] = useState<string | null>(null);
  const [liveEvent, setLiveEvent] = useState<(Partial<ActiveGame> & { game_id?: string; fen?: string; uci?: string; san?: string; ply?: number }) | null>(null);
  const [orientation, setOrientation] = useState<"white" | "black">("white");
  const [arenaPaused, setArenaPaused] = useState(false);
  const [replayPly, setReplayPly] = useState(0);
  const [autoplay, setAutoplay] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [mobileNav, setMobileNav] = useState(false);
  const [battle, setBattle] = useState({ model: "", opponent: "random", opponentVersion: "", games: 8, simulations: 32 });
  const initializedConfig = useRef(false);

  const refresh = useCallback(async () => {
    const results = await Promise.allSettled([
      json<TrainingStatus>("/api/status"), json<{ items: Game[] }>("/api/games?limit=100"),
      json<{ items: Metric[] }>("/api/metrics?limit=300"), json<{ items: Checkpoint[] }>("/api/checkpoints"),
      json<{ items: Evaluation[] }>("/api/evaluations"), json<{ items: { level: string; message: string; time: string }[] }>("/api/logs"),
    ]);
    if (results[0].status === "fulfilled") {
      setConnected(true); setStatus(results[0].value);
      if (!initializedConfig.current) { setConfig(results[0].value.config); initializedConfig.current = true; }
    } else setConnected(false);
    if (results[1].status === "fulfilled") {
      const refreshedGames = results[1].value.items;
      setGames(refreshedGames);
      setSelectedGame((current) => current ?? refreshedGames[0] ?? null);
    }
    if (results[2].status === "fulfilled") setMetrics(results[2].value.items);
    if (results[3].status === "fulfilled") setCheckpoints(results[3].value.items);
    if (results[4].status === "fulfilled") setEvaluations(results[4].value.items);
    if (results[5].status === "fulfilled") setLogs(results[5].value.items);
  }, []);

  useEffect(() => { const initial = window.setTimeout(refresh, 0); const timer = window.setInterval(refresh, 3000); return () => { window.clearTimeout(initial); window.clearInterval(timer); }; }, [refresh]);
  useEffect(() => {
    let socket: WebSocket | null = null; let retry: number | undefined; let disposed = false;
    const connect = () => {
      const url = `${API.replace(/^http/, "ws")}/ws/live`;
      socket = new WebSocket(url);
      socket.onmessage = (message) => {
        const event = JSON.parse(message.data);
        if (event.type === "move" && !arenaPaused) {
          if (!selectedActive || selectedActive === event.game_id) {
            setSelectedActive(event.game_id); setLiveEvent(event);
          }
        }
        if (event.type === "game_complete") refresh();
      };
      socket.onclose = () => { if (!disposed) retry = window.setTimeout(connect, 2500); };
    };
    connect();
    return () => { disposed = true; if (retry) clearTimeout(retry); socket?.close(); };
  }, [arenaPaused, refresh, selectedActive]);

  useEffect(() => {
    if (!autoplay || !selectedGame) return;
    const timer = window.setInterval(() => setReplayPly((ply) => {
      if (ply >= selectedGame.moves.length) { setAutoplay(false); return ply; }
      return ply + 1;
    }), 900 / speed);
    return () => clearInterval(timer);
  }, [autoplay, selectedGame, speed]);

  const act = async (path: string, body?: unknown) => {
    setBusy(true); setNotice(null);
    try {
      await json(path, { method: "POST", body: body ? JSON.stringify(body) : undefined });
      setNotice("Command accepted by the training service."); await refresh();
    } catch (error) { setNotice(error instanceof Error ? error.message : "Command failed"); }
    finally { setBusy(false); }
  };

  const active = status.active_games.find((game) => game.id === selectedActive) ?? status.active_games[0];
  const liveFen = liveEvent && liveEvent.game_id === active?.id ? liveEvent.fen : active?.fen;
  const liveMove = liveEvent && liveEvent.game_id === active?.id ? liveEvent.uci : active?.last_move;
  const replayFen = replayPly === 0 ? START_FEN : selectedGame?.moves[replayPly - 1]?.fen ?? selectedGame?.final_fen ?? START_FEN;

  const pageTitle = nav.find(([id]) => id === view)?.[1] ?? "Overview";
  const renderOverview = () => <>
    {!connected && <div className="connectionBanner"><WifiOff size={16} /><div><b>Training service is offline</b><span>Start the Python backend to see authoritative metrics and control training.</span></div></div>}
    <div className="metricGrid">
      <MetricCard label="COMPLETED GAMES" value={status.games_completed.toLocaleString()} note={`${status.active_games.length} active now`} icon={Database} />
      <MetricCard label="OPTIMIZER STEPS" value={status.step.toLocaleString()} note={`Replay buffer ${status.replay_size.toLocaleString()}`} icon={BrainCircuit} />
      <MetricCard label="POLICY LOSS" value={number(status.policy_loss)} note={`Value loss ${number(status.value_loss)}`} icon={Activity} />
      <MetricCard label="THROUGHPUT" value={`${status.games_per_minute.toFixed(2)}/m`} note={`${status.worker_count} scale workers · avg ${status.average_game_length.toFixed(1)} plies`} icon={Gauge} />
      <MetricCard label="CHAMPION MODEL" value={`v${status.model_version}`} note={`Candidate ${status.candidate_progress}/${status.candidate_target} · ${status.phase.replace("_", " ")}`} icon={ShieldCheck} />
      <MetricCard label="RUNTIME" value={duration(status.runtime_seconds)} note={`${status.device.toUpperCase()} device`} icon={Clock3} />
    </div>
    <div className="overviewGrid">
      <section className="panel livePanel">
        <div className="panelHead"><div><span className="kicker">LIVE SPECTATOR</span><h2>Watch the agents reason</h2></div><button className="textButton" onClick={() => setView("arena")}>Open arena <ChevronRight size={14} /></button></div>
        {active ? <div className="livePreview"><ChessBoard fen={liveFen} lastMove={liveMove} compact /><div className="liveInfo"><StatusPill connected status="running" /><h3>Game {active.id}</h3><p>Model v{active.model_version} self-play · Iteration {active.iteration}</p><div className="moveStrip">{active.moves?.slice(-8).map((move) => <span key={move.ply}>{move.san}</span>)}</div></div></div> : <Empty icon={MonitorDot} title="No self-play game is active" copy="Start training and the next real game will stream here." />}
      </section>
      <section className="panel systemPanel">
        <div className="panelHead"><div><span className="kicker">SYSTEM HEALTH</span><h2>Compute resources</h2></div><Cpu size={18} /></div>
        {[
          { label: "CPU", bar: status.system.cpu_percent, value: `${status.system.cpu_percent.toFixed(1)}%` },
          { label: "RAM", bar: status.system.ram_percent, value: `${status.system.ram_percent.toFixed(1)}%` },
          { label: "PROCESS", bar: Math.min(100, status.system.process_mb / 20), value: `${status.system.process_mb.toFixed(0)} MB` },
        ].map((resource) => <div className="resource" key={resource.label}><div><span>{resource.label}</span><b>{resource.value}</b></div><i><span style={{ width: `${resource.bar}%` }} /></i></div>)}
        <div className="resultSplit"><div><b>{percent(status.white_win_rate)}</b><span>White wins</span></div><div><b>{percent(status.draw_rate)}</b><span>Draws</span></div><div><b>{percent(status.black_win_rate)}</b><span>Black wins</span></div></div>
      </section>
    </div>
    <section className="panel recentPanel"><div className="panelHead"><div><span className="kicker">RECENT SELF-PLAY</span><h2>Completed games</h2></div><button className="textButton" onClick={() => setView("history")}>View history <ChevronRight size={14} /></button></div><GameTable games={games.slice(0, 6)} onSelect={(game) => { setSelectedGame(game); setReplayPly(0); setView("history"); }} /></section>
  </>;

  const renderArena = () => <div className="arenaLayout">
    <aside className="activeList panel"><div className="panelHead"><div><span className="kicker">WORKERS</span><h2>Active games</h2></div><span className="countBadge">{status.active_games.length}</span></div>
      {status.active_games.length ? status.active_games.map((game) => <button key={game.id} className={`gameChoice ${active?.id === game.id ? "selected" : ""}`} onClick={() => { setSelectedActive(game.id); setLiveEvent(null); }}><span className="miniBoard"><ChessBoard fen={game.fen} lastMove={game.last_move} compact /></span><span><b>Game {game.id}</b><small>v{game.model_version} · {game.moves.length} plies</small></span><i /></button>) : <Empty icon={ListTree} title="No workers playing" copy="Active self-play workers appear here automatically." />}
    </aside>
    <section className="panel arenaMain"><div className="arenaToolbar"><div><span className="kicker">{active ? `GAME ${active.id}` : "LIVE ARENA"}</span><h2>{active ? `Model v${active.model_version} vs Model v${active.model_version}` : "Waiting for a game"}</h2></div><div className="toolbarGroup"><button className="iconButton" title="Flip board" onClick={() => setOrientation((o) => o === "white" ? "black" : "white")}><RotateCcw size={16} /></button><button className={`iconButton ${arenaPaused ? "active" : ""}`} title="Pause visual playback" onClick={() => setArenaPaused(!arenaPaused)}>{arenaPaused ? <Play size={16} /> : <Pause size={16} />}</button></div></div>
      <div className="arenaStage"><div className="playerRow"><span className="agentAvatar black">B</span><div><b>Model v{active?.model_version ?? status.model_version}</b><small>Black agent</small></div><span>{liveFen?.split(" ")[1] === "b" ? "THINKING" : "WAITING"}</span></div><ChessBoard fen={liveFen} orientation={orientation} lastMove={liveMove} /><div className="playerRow"><span className="agentAvatar white">W</span><div><b>Model v{active?.model_version ?? status.model_version}</b><small>White agent</small></div><span>{liveFen?.split(" ")[1] === "w" ? "THINKING" : "WAITING"}</span></div></div>
    </section>
    <aside className="panel movePanel"><div className="panelHead"><div><span className="kicker">MOVE STREAM</span><h2>Search decisions</h2></div>{arenaPaused ? <CirclePause size={18} /> : <CirclePlay size={18} />}</div>{active ? <div className="moveList">{active.moves.map((move) => <button key={move.ply}><span>{move.ply}.</span><b>{move.san}</b><small>v {move.root_value?.toFixed(2)} · {move.visits} visits</small></button>)}</div> : <Empty icon={Activity} title="No moves yet" copy="Moves are sampled only for the game you are watching." />}</aside>
  </div>;

  const renderHistory = () => <div className="historyLayout"><section className="panel historyList"><div className="panelHead"><div><span className="kicker">DURABLE ARCHIVE</span><h2>{games.length} stored games</h2></div><History size={18} /></div><GameTable games={games} activeId={selectedGame?.id} onSelect={(game) => { setSelectedGame(game); setReplayPly(0); setAutoplay(false); }} /></section><section className="panel replayPanel">{selectedGame ? <><div className="replayTitle"><div><span className="kicker">GAME {selectedGame.id}</span><h2>{selectedGame.white_agent} vs {selectedGame.black_agent}</h2><p>{selectedGame.result} · {selectedGame.termination}</p></div><a className="iconButton" href={`${API}/api/games/${selectedGame.id}/pgn`} title="Download PGN"><Download size={16} /></a></div><div className="replayBody"><div><ChessBoard fen={replayFen} orientation={orientation} lastMove={replayPly ? selectedGame.moves[replayPly - 1]?.uci : undefined} /><input className="timeline" type="range" min="0" max={selectedGame.moves.length} value={replayPly} onChange={(event) => setReplayPly(Number(event.target.value))} /><div className="replayControls"><button onClick={() => setReplayPly(0)}><ChevronFirst /></button><button onClick={() => setReplayPly(Math.max(0, replayPly - 1))}><ChevronLeft /></button><button className="primaryRound" onClick={() => setAutoplay(!autoplay)}>{autoplay ? <Pause /> : <Play />}</button><button onClick={() => setReplayPly(Math.min(selectedGame.moves.length, replayPly + 1))}><ChevronRight /></button><button onClick={() => setReplayPly(selectedGame.moves.length)}><ChevronLast /></button><select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option></select><button onClick={() => setOrientation((o) => o === "white" ? "black" : "white")}><RotateCcw /></button></div></div><div className="replayMoves"><div className="fenBox"><span>FEN AFTER PLY {replayPly}</span><code>{replayFen}</code></div><div className="notationGrid">{selectedGame.moves.map((move) => <button key={move.ply} className={replayPly === move.ply ? "active" : ""} onClick={() => setReplayPly(move.ply)}><span>{move.ply}.</span>{move.san}</button>)}</div></div></div></> : <Empty icon={History} title="Choose a stored game" copy="The replay viewer uses the exact FEN sequence persisted by the backend." />}</section></div>;

  const renderAnalytics = () => <div className="analyticsGrid"><section className="panel chartPanel wide"><div className="panelHead"><div><span className="kicker">OPTIMIZATION SIGNAL</span><h2>Training losses</h2></div><span className="legendNote">Real optimizer steps only</span></div>{metrics.length ? <TrainingLossChart metrics={metrics} /> : <Empty icon={BarChart3} title="No optimization data yet" copy="Loss curves appear after the first self-play batch is optimized." />}</section><section className="panel chartPanel"><div className="panelHead"><div><span className="kicker">THROUGHPUT</span><h2>Games per minute</h2></div><Gauge size={18} /></div>{metrics.length ? <ThroughputChart metrics={metrics} /> : <Empty icon={Gauge} title="Awaiting samples" copy="Throughput is measured from actual iteration runtime." />}</section><section className="panel evaluationPanel"><div className="panelHead"><div><span className="kicker">PLAYING STRENGTH</span><h2>Evaluation protocol</h2></div><Trophy size={18} /></div>{evaluations.length ? evaluations.slice(0, 5).map((item) => <div className="evaluationRow" key={item.id}><div><b>v{item.model_a} vs {item.model_b}</b><small>{item.games} color-balanced games</small></div><div className="wdl"><span className="win">{item.wins}W</span><span>{item.draws}D</span><span className="loss">{item.losses}L</span></div></div>) : <Empty icon={Trophy} title="No evaluations completed" copy="Use Model Battle to compare checkpoints or fixed baselines." />}</section></div>;

  const renderModels = () => <section className="panel tablePanel"><div className="panelHead"><div><span className="kicker">PROMOTED MODEL REGISTRY</span><h2>Champion checkpoints</h2></div><span className="countBadge">{checkpoints.length}</span></div>{checkpoints.length ? <div className="dataTable"><div className="tableHeader"><span>Version</span><span>Created</span><span>Training state</span><span>Architecture</span><span>Promotion</span><span>Actions</span></div>{checkpoints.map((checkpoint) => <div className="tableRow" key={checkpoint.version}><span><b>Model v{checkpoint.version}</b></span><span>{new Date(checkpoint.created_at).toLocaleString()}</span><span>Iter {checkpoint.iteration} · {checkpoint.games} games</span><span>{checkpoint.architecture.channels}ch · {checkpoint.architecture.residual_blocks} blocks</span><span>{checkpoint.evaluation.score != null ? `${(checkpoint.evaluation.score * 100).toFixed(1)}% score` : bytes(checkpoint.file_size)}</span><span className="rowActions"><button onClick={() => act("/api/training/start", { resume_version: checkpoint.version })} title="Resume from checkpoint"><Play size={14} /></button><a href={`${API}/api/checkpoints/${checkpoint.version}/download`} title="Download checkpoint"><Download size={14} /></a></span></div>)}</div> : <Empty icon={Archive} title="No promoted model yet" copy="A candidate becomes a versioned model only after completing its cycle and passing the promotion match." />}</section>;

  const renderBattle = () => <div className="battleGrid"><section className="panel battleSetup"><div className="panelHead"><div><span className="kicker">CONTROLLED EVALUATION</span><h2>Configure a model battle</h2></div><Swords size={19} /></div><label>Candidate model<select value={battle.model} onChange={(e) => setBattle({ ...battle, model: e.target.value })}><option value="">Current v{status.model_version}</option>{checkpoints.map((checkpoint) => <option key={checkpoint.version} value={checkpoint.version}>Model v{checkpoint.version}</option>)}</select></label><label>Opponent<select value={battle.opponent} onChange={(e) => setBattle({ ...battle, opponent: e.target.value })}><option value="random">Random legal baseline</option><option value="heuristic">Capture-first heuristic</option><option value="checkpoint">Previous checkpoint</option></select></label>{battle.opponent === "checkpoint" && <label>Opponent version<select value={battle.opponentVersion} onChange={(e) => setBattle({ ...battle, opponentVersion: e.target.value })}><option value="">Choose a checkpoint</option>{checkpoints.map((checkpoint) => <option key={checkpoint.version} value={checkpoint.version}>Model v{checkpoint.version}</option>)}</select></label>}<label>Games<input type="number" min="2" max="100" value={battle.games} onChange={(e) => setBattle({ ...battle, games: Number(e.target.value) })} /></label><label>MCTS simulations / move<input type="number" min="4" max="1000" value={battle.simulations} onChange={(e) => setBattle({ ...battle, simulations: Number(e.target.value) })} /></label><div className="protocol"><Check size={15} /><span>Paired games share a short randomized opening, colors alternate, and repetition is discouraged.</span></div><button className="primaryButton" disabled={!connected || busy} onClick={() => act("/api/evaluations", { model_version: battle.model ? Number(battle.model) : null, opponent: battle.opponent, opponent_version: battle.opponentVersion ? Number(battle.opponentVersion) : null, games: battle.games, simulations: battle.simulations })}>{busy ? <LoaderCircle className="spin" size={16} /> : <Swords size={16} />} Start evaluation</button></section><section className="panel battleResults"><div className="panelHead"><div><span className="kicker">MATCH ARCHIVE</span><h2>Comparative results</h2></div><Trophy size={18} /></div>{evaluations.length ? evaluations.map((item) => <div className="battleResult" key={item.id}><div className="battleNames"><b>Model v{item.model_a}</b><span>VS</span><b>{item.model_b.startsWith("v") ? `Model ${item.model_b}` : item.model_b}</b></div><div className="battleScore"><strong>{item.wins}</strong><span>{item.draws} draws</span><strong>{item.losses}</strong></div><div className="battleBar"><i style={{ width: `${item.wins / item.games * 100}%` }} /><em style={{ width: `${item.draws / item.games * 100}%` }} /></div><small>{new Date(item.created_at).toLocaleString()} · {item.games} games</small></div>) : <Empty icon={Swords} title="No battles completed" copy="Run a real match to compare two checkpoints or a fixed baseline." />}</section></div>;

  const renderConfig = () => <div className="configGrid"><section className="panel configPanel"><div className="panelHead"><div><span className="kicker">SELF-PLAY</span><h2>Game generation</h2></div><FlaskConical size={18} /></div><ConfigInput label="Experiment name" field="experiment_name" value={config.experiment_name} setConfig={setConfig} /><ConfigInput label="Games / iteration" field="games_per_iteration" value={config.games_per_iteration} setConfig={setConfig} /><ConfigInput label="Parallel workers" field="parallel_workers" value={config.parallel_workers} setConfig={setConfig} /><ConfigInput label="MCTS simulations" field="mcts_simulations" value={config.mcts_simulations} setConfig={setConfig} /><ConfigInput label="Maximum full moves" field="max_game_moves" value={config.max_game_moves} setConfig={setConfig} /><ConfigInput label="Random seed" field="random_seed" value={config.random_seed} setConfig={setConfig} /></section><section className="panel configPanel"><div className="panelHead"><div><span className="kicker">OPTIMIZER</span><h2>Learning schedule</h2></div><BrainCircuit size={18} /></div><ConfigInput label="Batch size" field="batch_size" value={config.batch_size} setConfig={setConfig} /><ConfigInput label="Learning rate" field="learning_rate" value={config.learning_rate} setConfig={setConfig} step="0.0001" /><ConfigInput label="Replay capacity" field="replay_capacity" value={config.replay_capacity} setConfig={setConfig} /><ConfigInput label="Updates / iteration" field="training_updates" value={config.training_updates} setConfig={setConfig} /><ConfigInput label="Network channels" field="channels" value={config.channels} setConfig={setConfig} /><ConfigInput label="Residual blocks" field="residual_blocks" value={config.residual_blocks} setConfig={setConfig} /></section><section className="panel configPanel promotionPanel"><div className="panelHead"><div><span className="kicker">MODEL PROMOTION</span><h2>Candidate quality gate</h2></div><ShieldCheck size={18} /></div><ConfigInput label="Iterations / candidate" field="checkpoint_interval" value={config.checkpoint_interval} setConfig={setConfig} /><ConfigInput label="Promotion match games" field="promotion_games" value={config.promotion_games} setConfig={setConfig} /><ConfigInput label="Required score" field="promotion_threshold" value={config.promotion_threshold} setConfig={setConfig} step="0.01" /><ConfigInput label="Evaluation MCTS" field="evaluation_simulations" value={config.evaluation_simulations} setConfig={setConfig} /><ConfigInput label="Opening diversity plies" field="evaluation_opening_plies" value={config.evaluation_opening_plies} setConfig={setConfig} /><ConfigToggle label="Stop after a new model is promoted" copy="Rejected candidates keep training; a successfully saved checkpoint stops the session." checked={config.stop_after_promotion} onChange={(checked) => setConfig((current) => ({ ...current, stop_after_promotion: checked }))} /><ConfigInput label="Autosave interval" field="autosave_interval" value={config.autosave_interval} setConfig={setConfig} /></section><section className="panel promotionStatus"><span className="kicker">CURRENT CANDIDATE</span><strong>{status.candidate_progress} / {status.candidate_target} iterations</strong><p>{status.last_promotion ? `Last gate: ${status.last_promotion.status} at ${(status.last_promotion.score * 100).toFixed(1)}% score.` : "The next version is created only after this cycle passes its promotion match."}</p><small>{config.stop_after_promotion ? "Auto-stop is armed for the next successful promotion." : "Training will continue after successful promotions."}</small></section><section className="panel launchPanel"><div><span className="kicker">TRAINING SESSION</span><h2>Launch this configuration</h2><p>Training autosaves into one rolling candidate file. A numbered model is created only after the candidate completes its cycle and beats the champion.</p></div><div className="launchSummary"><span><b>{config.games_per_iteration * config.checkpoint_interval}</b> games / gate</span><span><b>{config.evaluation_simulations}</b> eval sims</span><span><b>{config.stop_after_promotion ? "STOP" : "LOOP"}</b> after promotion</span></div><button className="primaryButton" disabled={!connected || busy || ["running", "paused"].includes(status.status)} onClick={() => act("/api/training/start", { config })}>{busy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />} Start candidate training</button></section></div>;

  const renderLogs = () => <section className="panel logPanel"><div className="panelHead"><div><span className="kicker">STRUCTURED RUNTIME LOG</span><h2>Training service events</h2></div><span className="countBadge">{logs.length}</span></div>{logs.length ? <div className="logList">{logs.map((log, index) => <div className={`logLine ${log.level}`} key={`${log.time}-${index}`}><time>{new Date(log.time).toLocaleTimeString()}</time><span>{log.level.toUpperCase()}</span><p>{log.message}</p></div>)}</div> : <Empty icon={TerminalSquare} title="No runtime events" copy="Training and checkpoint events are logged here by the backend." />}</section>;

  const render = () => ({ overview: renderOverview, arena: renderArena, history: renderHistory, analytics: renderAnalytics, models: renderModels, battle: renderBattle, config: renderConfig, logs: renderLogs })[view]();

  return <main className="appShell">
    <aside className={`sidebar ${mobileNav ? "open" : ""}`}><div className="brand"><span className="brandMark">C</span><div><b>CARO LAB</b><small>SELF-LEARNING CHESS</small></div><button className="closeNav" onClick={() => setMobileNav(false)}><X size={18} /></button></div><nav>{nav.map(([id, label, Icon]) => <button key={id} className={view === id ? "active" : ""} onClick={() => { setView(id); setMobileNav(false); }}><Icon size={17} /><span>{label}</span></button>)}</nav><div className="sidebarStatus"><StatusPill connected={connected} status={status.status} /><span>Model v{status.model_version} · Iter {status.iteration}</span></div></aside>
    <section className="workspace"><header className="topbar"><button className="menuButton" onClick={() => setMobileNav(true)}><Menu size={19} /></button><div><span className="breadcrumb">EXPERIMENT / {config.experiment_name.toUpperCase()}</span><h1>{pageTitle}</h1></div><div className="topActions"><div className="runtime"><span>RUNTIME</span><b>{duration(status.runtime_seconds)}</b></div>{status.status === "running" && <button className="controlButton" onClick={() => act("/api/training/pause")}><Pause size={14} /> Pause</button>}{status.status === "paused" && <button className="controlButton" onClick={() => act("/api/training/resume")}><Play size={14} /> Resume</button>}{["running", "paused", "stopping"].includes(status.status) && <button className="stopButton" onClick={() => act("/api/training/stop")}>Stop</button>}<StatusPill connected={connected} status={status.status} /></div></header>{notice && <div className="notice"><span>{notice}</span><button onClick={() => setNotice(null)}><X size={14} /></button></div>}<div className="pageContent">{render()}</div></section>
  </main>;
}

function GameTable({ games, onSelect, activeId }: { games: Game[]; onSelect: (game: Game) => void; activeId?: string }) {
  if (!games.length) return <Empty icon={Database} title="No completed games" copy="The archive will populate from real self-play results." />;
  return <div className="gameTable"><div className="gameTableHead"><span>Game</span><span>Agents</span><span>Result</span><span>Length</span><span>Termination</span><span>Completed</span></div>{games.map((game) => <button key={game.id} className={activeId === game.id ? "active" : ""} onClick={() => onSelect(game)}><span><b>#{game.id}</b><small>Iteration {game.iteration}</small></span><span>v{game.model_version} ↔ v{game.model_version}</span><span className={`result ${game.result === "1/2-1/2" ? "draw" : "win"}`}>{game.result}</span><span>{game.move_count} plies</span><span>{game.termination}</span><span>{new Date(game.created_at).toLocaleString()}</span></button>)}</div>;
}

function ConfigInput({ label, field, value, setConfig, step }: { label: string; field: keyof TrainingConfig; value: string | number; setConfig: React.Dispatch<React.SetStateAction<TrainingConfig>>; step?: string }) {
  return <label className="configInput"><span>{label}</span><input type={typeof value === "number" ? "number" : "text"} step={step} value={value} onChange={(event) => setConfig((current) => ({ ...current, [field]: typeof value === "number" ? Number(event.target.value) : event.target.value }))} /></label>;
}

function ConfigToggle({ label, copy, checked, onChange }: { label: string; copy: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return <label className="configToggle"><span><b>{label}</b><small>{copy}</small></span><input aria-label={label} type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /></label>;
}
