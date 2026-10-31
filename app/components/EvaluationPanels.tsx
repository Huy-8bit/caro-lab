"use client";

import { AlertTriangle, CheckCircle2, CircleSlash, LoaderCircle, PauseCircle } from "lucide-react";
import type { Checkpoint, Evaluation, EvaluationProgress, QualityAlert } from "../types";

function clock(seconds: number | null | undefined) {
  if (seconds == null) return "—";
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const secs = (total % 60).toString().padStart(2, "0");
  return minutes >= 60 ? `${Math.floor(minutes / 60)}h ${(minutes % 60).toString().padStart(2, "0")}m` : `${minutes}:${secs}`;
}

const pct = (value: number | null | undefined) => (value == null ? "—" : `${(value * 100).toFixed(1)}%`);

const STATUS_LABEL: Record<string, string> = {
  running: "Running", promoted: "Promoted", rejected: "Rejected", complete: "Complete", interrupted: "Interrupted", error: "Error",
};

export function StatusTag({ status }: { status: string }) {
  const Icon = status === "running" ? LoaderCircle : status === "promoted" || status === "complete" ? CheckCircle2 : status === "interrupted" ? PauseCircle : CircleSlash;
  return <span className={`statusTag ${status}`}><Icon size={12} className={status === "running" ? "spin" : undefined} />{STATUS_LABEL[status] ?? status}</span>;
}

export function EvaluationProgressPanel({ progress, compact = false }: { progress: EvaluationProgress | null; compact?: boolean }) {
  if (!progress) return null;
  const done = progress.games_completed;
  const total = Math.max(1, progress.games_total);
  const share = (count: number) => `${count / total * 100}%`;
  const terminations = Object.entries(progress.terminations).sort((a, b) => b[1] - a[1]);
  return <section className={`panel evalProgress ${compact ? "compact" : ""}`} aria-live="polite">
    <div className="panelHead">
      <div>
        <span className="kicker">{progress.kind === "promotion" ? "PROMOTION MATCH" : "MODEL BATTLE"}</span>
        <h2>{progress.candidate_label} <span className="vs">vs</span> {progress.opponent_label}</h2>
      </div>
      <StatusTag status={progress.status} />
    </div>
    <div className="evalBody">
      <div className="evalCounts">
        <strong>{done}<small> / {progress.games_total} games</small></strong>
        <span><b className="win">{progress.wins}</b> W · <b>{progress.draws}</b> D · <b className="loss">{progress.losses}</b> L</span>
        <span>Score <b>{pct(progress.score)}</b></span>
      </div>
      <div className="evalBar" role="progressbar" aria-valuemin={0} aria-valuemax={progress.games_total} aria-valuenow={done} aria-label="Evaluation games completed">
        <i className="win" style={{ width: share(progress.wins) }} />
        <i className="draw" style={{ width: share(progress.draws) }} />
        <i className="loss" style={{ width: share(progress.losses) }} />
      </div>
      <div className="evalLegend"><span><i className="win" />Candidate wins</span><span><i className="draw" />Draws</span><span><i className="loss" />Candidate losses</span><span><i className="rest" />Not played yet</span></div>
      <dl className="evalTimes">
        <div><dt>Elapsed</dt><dd>{clock(progress.elapsed_seconds)}</dd></div>
        <div><dt>ETA</dt><dd>{progress.status === "running" ? (done ? clock(progress.eta_seconds) : "after first game") : "—"}</dd></div>
        <div><dt>In progress</dt><dd>{progress.current_games.length ? progress.current_games.map((game) => `#${game.game} (${game.candidate_color}, ${game.plies} plies)`).join(" · ") : "—"}</dd></div>
      </dl>
      {!compact && <div className="evalTerminations">
        <span className="kicker">TERMINATIONS</span>
        {terminations.length ? terminations.map(([name, count]) => <span key={name}><b>{count}</b> {name}</span>) : <span>No finished games yet</span>}
      </div>}
      {progress.decision?.reasons && progress.decision.reasons.length > 0 && <ul className="evalReasons">
        {progress.decision.reasons.map((reason) => <li key={reason}>{reason}</li>)}
        {progress.decision.action && <li className="muted">{progress.decision.action}</li>}
      </ul>}
    </div>
  </section>;
}

export function QualityAlerts({ alerts }: { alerts: QualityAlert[] | undefined }) {
  if (!alerts?.length) return null;
  return <div className="qualityAlerts" role="alert">
    {alerts.map((alert) => <div key={alert.code}><AlertTriangle size={15} /><span><b>Self-play quality warning</b>{alert.message}</span></div>)}
  </div>;
}

export function promotionLabel(evaluation: Checkpoint["evaluation"]) {
  if (evaluation.status === "bootstrap") return "Bootstrap · no match played";
  if (evaluation.score != null) return `Beat ${evaluation.opponent ?? "champion"} · ${(evaluation.score * 100).toFixed(1)}%`;
  return "—";
}

export function evaluationTitle(item: Evaluation) {
  if (item.title) return item.title;
  return `Model v${item.model_a} vs ${item.model_b.startsWith("v") ? `Model ${item.model_b}` : item.model_b}`;
}
