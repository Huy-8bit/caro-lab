export type TrainingConfig = {
  experiment_name: string;
  games_per_iteration: number;
  parallel_workers: number;
  mcts_simulations: number;
  max_game_moves: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  replay_capacity: number;
  training_updates: number;
  checkpoint_interval: number;
  evaluation_frequency: number;
  temperature_moves: number;
  temperature: number;
  dirichlet_alpha: number;
  dirichlet_epsilon: number;
  c_puct: number;
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
  root_value: number;
  visits: number;
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
  throughput: number;
  created_at: string;
};

export type Checkpoint = {
  version: number;
  path: string;
  created_at: string;
  iteration: number;
  games: number;
  architecture: { channels: number; residual_blocks: number; policy_size: number };
  config: TrainingConfig;
  evaluation: Record<string, number>;
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
};

export type TrainingStatus = {
  status: "idle" | "running" | "paused" | "stopping" | "error";
  iteration: number;
  step: number;
  model_version: number;
  games_completed: number;
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
