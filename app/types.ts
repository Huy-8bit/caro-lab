export type TrainingConfig = {
  experiment_name: string;
  games_per_iteration: number;
  parallel_workers: number;
  self_play_mode: "local" | "distributed";
  worker_sync_interval: number;
  worker_job_timeout: number;
  mcts_simulations: number;
  max_game_moves: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  replay_capacity: number;
  training_updates: number;
  checkpoint_interval: number;
  promotion_games: number;
  promotion_threshold: number;
  evaluation_simulations: number;
  evaluation_opening_plies: number;
  gate_max_repetition_rate: number;
  gate_min_decisive_rate: number;
  reject_reset_score: number;
  stop_after_promotion: boolean;
  autosave_interval: number;
  temperature_moves: number;
  temperature: number;
  temperature_final: number;
  selfplay_opening_plies: number;
  endgame_curriculum_fraction: number;
  dirichlet_alpha: number;
  dirichlet_epsilon: number;
  c_puct: number;
  policy_target: "completed_q" | "visits";
  completed_q_scale: number;
  draw_contempt: number;
  draw_sample_weight: number;
  repetition_sample_weight: number;
  adjudicated_sample_weight: number;
  sample_length_reference: number;
  replay_max_draw_fraction: number;
  analytics_window: number;
  random_seed: number;
  device: string;
  channels: number;
  residual_blocks: number;
  history_retention: number;
  event_sample_rate: number;
};

export type MoveRecord = {
  ply: number;
  uci: string;
  san: string;
  fen_before: string;
  fen: string;
  root_value: number | null;
  visits: number;
  opening?: boolean;
};

export type Game = {
  id: string;
  iteration: number;
  model_version: number;
  white_agent: string;
  black_agent: string;
  result: string;
  winner: string | null;
  move_count: number;
  duration_seconds: number;
  created_at: string;
  pgn: string;
  final_fen: string;
  termination: string;
  moves: MoveRecord[];
};

export type ActiveGame = {
  id: string;
  iteration: number;
  model_version: number;
  status: string;
  fen?: string;
  last_move?: string;
  moves: MoveRecord[];
};

export type Metric = {
  id: number;
  iteration: number;
  step: number;
  policy_loss: number;
  value_loss: number;
  total_loss: number;
  games_completed: number;
  /** Games per minute of wall-clock iteration time, including the optimizer. */
  throughput: number;
  /** Games per minute of the self-play phase only. */
  selfplay_games_per_minute?: number | null;
  selfplay_seconds?: number | null;
  optimize_seconds?: number | null;
  replay_draw_share?: number | null;
  created_at: string;
};

export type Rates = {
  draw_rate: number | null;
  repetition_rate: number | null;
  checkmate_rate: number | null;
  decisive_rate: number | null;
  adjudicated_rate: number | null;
};

export type IterationQuality = Rates & {
  iteration: number;
  games: number;
  draws: number;
  repetitions: number;
  checkmates: number;
  adjudicated: number;
  average_plies: number | null;
  total_loss: number | null;
  rolling: Rates & { games: number; total_loss: number | null; curriculum_games?: number; curriculum_checkmate_rate?: number | null };
};

export type QualityAlert = { level: string; code: string; message: string };

export type Analytics = { window: number; iterations: IterationQuality[]; rolling: IterationQuality["rolling"]; alerts: QualityAlert[] };

export type MatchSummary = {
  score: number;
  wins: number;
  draws: number;
  losses: number;
  games: number;
  repetition_rate?: number;
  decisive_rate?: number;
  checkmate_rate?: number;
  termination_counts?: Record<string, number>;
  elo_diff?: number | null;
  elo_low?: number | null;
  elo_high?: number | null;
};

export type EvaluationProgress = {
  id: string;
  kind: "promotion" | "manual";
  status: "running" | "promoted" | "rejected" | "complete" | "interrupted" | "error";
  title: string;
  candidate_label: string;
  opponent_label: string;
  games_total: number;
  games_completed: number;
  wins: number;
  draws: number;
  losses: number;
  score: number | null;
  terminations: Record<string, number>;
  current_games: { game_index: number; game: number; candidate_color: string; plies: number; fen: string | null; worker_id: string | null }[];
  started_at: string;
  elapsed_seconds: number;
  eta_seconds: number | null;
  decision: (Partial<MatchSummary> & { status: string; reasons?: string[]; action?: string }) | null;
};

export type Checkpoint = {
  version: number;
  path: string;
  created_at: string;
  iteration: number;
  games: number;
  architecture: { channels: number; residual_blocks: number; policy_size: number };
  config: TrainingConfig;
  evaluation: Partial<MatchSummary> & {
    status?: "bootstrap" | "promoted" | string;
    opponent?: string | null;
    note?: string;
    threshold?: number;
  };
  file_size: number;
};

export type Evaluation = {
  id: string;
  model_a: number;
  model_b: string;
  games: number;
  wins: number;
  draws: number;
  losses: number;
  created_at: string;
  kind?: "promotion" | "manual";
  status?: string;
  title?: string;
  reasons?: string[];
  action?: string;
  score?: number;
  repetition_rate?: number;
  decisive_rate?: number;
  elo_diff?: number | null;
  elo_low?: number | null;
  elo_high?: number | null;
  termination_counts?: Record<string, number>;
  details?: { termination?: string; opening_plies?: number; result?: string; model_color?: string }[];
};

export type TrainingStatus = {
  status: "idle" | "running" | "paused" | "stopping" | "error";
  phase: "idle" | "self_play" | "optimizing" | "evaluating" | "waiting_for_workers";
  iteration: number;
  step: number;
  model_version: number;
  candidate_version: number;
  candidate_progress: number;
  candidate_target: number;
  last_promotion: (Partial<MatchSummary> & {
    status: "promoted" | "rejected" | "bootstrap";
    version: number;
    candidate_version?: number;
    champion_version?: number;
    reasons?: string[];
    action?: string;
  }) | null;
  evaluation: EvaluationProgress | null;
  jobs?: { queued: number; leased: number; complete: number };
  quality?: { rolling: Analytics["rolling"]; alerts: QualityAlert[]; window: number };
  replay?: { size: number; counts: Record<string, number>; sampling_share: Record<string, number>; max_draw_fraction: number };
  selfplay_games_per_minute?: number | null;
  selfplay_seconds?: number | null;
  optimize_seconds?: number | null;
  games_completed: number;
  worker_count: number;
  workers: { id: string; status: string; games_completed: number; revision: number | null; job_id?: string | null; last_seen: string }[];
  active_games: ActiveGame[];
  runtime_seconds: number;
  games_per_minute: number;
  average_game_length: number;
  white_win_rate: number;
  black_win_rate: number;
  draw_rate: number;
  policy_loss: number | null;
  value_loss: number | null;
  total_loss: number | null;
  replay_size: number;
  device: string;
  system: { cpu_percent: number; ram_percent: number; process_mb: number; platform: string };
  last_error: string | null;
  config: TrainingConfig;
};
