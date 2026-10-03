// ─── Standard API envelope ───────────────────────────────────────────────────

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  message: string;
  error_code?: string;
}

export interface ApiErrorResponse {
  success: false;
  message: string;
  error_code: string;
  status?: number;
}

// ─── Dashboard Types ─────────────────────────────────────────────────────────
export interface DashboardKPIs {
  total_machines: number;
  healthy_machines: number;
  warning_machines: number;
  critical_machines: number;
  model_f1_score: number;
  downtime_prevented_hrs: number;
  cost_saved_usd: number;
  total_predictions_logged: number;
}

export interface DashboardData {
  kpis: DashboardKPIs;
  current_dataset: string;
  current_model: string;
}

export interface DashboardResponse extends ApiResponse<DashboardData> {}

// ─── Machine Types ───────────────────────────────────────────────────────────
export interface Machine {
  machineID: number;
  model: string;
  age: number;
  status: 'Healthy' | 'Warning' | 'Critical';
}

export interface MachinesResponse extends ApiResponse<{
  machines: Machine[];
}> {}

// ─── Prediction Types ────────────────────────────────────────────────────────
export interface PredictionResult {
  prediction_id?: string;
  prediction: number;
  probability: number;
  confidence: number;
  risk_level: string;
  recommended_action: string;
  latency_ms?: number;
  model_version?: string;
  explanation_available: boolean;
}

export interface PredictionResponse extends ApiResponse<PredictionResult> {}

// ─── Model Types ─────────────────────────────────────────────────────────────
export interface ModelInfo {
  active_model: string;
  version: string;
  algorithm: string;
  training_date: string;
  feature_count: number;
  status: string;
  model_ready: boolean;
}

export interface ModelData {
  model: ModelInfo;
}

export interface ModelResponse extends ApiResponse<ModelData> {}

// ─── Health / System Status Types ────────────────────────────────────────────
export type ArtifactStatus = Record<string, boolean>;

export interface HealthData {
  status: 'healthy' | 'degraded';
  backend: string;
  backend_status?: string;
  model_loaded: boolean;
  artifacts: ArtifactStatus;
  database_connected: boolean;
  mlflow_available: boolean;
  mlflow_connected?: boolean;
  uptime_seconds: number;
  api_health: 'Optimal' | 'Degraded';
  overall_health_score?: number;
  cpu_percent?: number;
  memory_percent?: number;
  disk_percent?: number;
  prediction_count?: number;
  failure_count?: number;
}

export interface HealthResponse extends ApiResponse<HealthData> {}

// ─── Executive Monitoring Summary Type ──────────────────────────────────────
export interface MonitoringData {
  system_status: string;
  model_status: string;
  overall_health_score: number;
  backend_status: string;
  model_loaded: boolean;
  mlflow_connected: boolean;
  database_connected: boolean;
  uptime_seconds: number;

  cpu_percent: number;
  memory_percent: number;
  memory_used_mb: number;
  disk_percent: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  throughput_rpm: number;

  prediction_count: number;
  failure_count: number;
  prediction_success_rate: number;
  prediction_failure_rate: number;

  data_drift: {
    detected: boolean;
    drift_score: number;
    number_of_features: number;
    number_of_drifted_features: number;
    share_of_drifted_features: number;
    dataset_drift: boolean;
    feature_metrics: Record<string, {
      drift_detected: boolean;
      p_value: number;
      drift_score: number;
      ref_mean?: number;
      cur_mean?: number;
      stat_test: string;
    }>;
  };
  model_drift: {
    detected: boolean;
    performance_degradation: boolean;
    confidence_degradation: boolean;
    distribution_change: boolean;
    training_vs_production_kl_divergence: number;
    avg_confidence_drop: number;
  };
  data_quality: {
    total_rows: number;
    missing_values_count: number;
    missing_values_share: number;
    duplicate_rows_count: number;
    empty_columns_count: number;
  };

  current_model: string;
  model_version: string;
  last_training_time: string;
  mlflow_experiment: string;
  mlflow_run_id: string;
}

export interface MonitoringResponse extends ApiResponse<MonitoringData> {}

// ─── Alert Type ──────────────────────────────────────────────────────────────
export interface AlertItem {
  id: string;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  category: string;
  title: string;
  message: string;
  timestamp: string;
  action: string;
  acknowledged?: boolean;
}

export interface AlertsResponse extends ApiResponse<{
  total_alerts: number;
  critical_count: number;
  warning_count: number;
  alerts: AlertItem[];
}> {}

// ─── Prediction Log Entry ────────────────────────────────────────────────────
export interface PredictionLogItem {
  id: number;
  prediction_id: string;
  timestamp: string;
  machine_id: number;
  volt: number;
  rotate: number;
  pressure: number;
  vibration: number;
  prediction: number;
  probability: number;
  confidence: number;
  latency_ms: number;
  model_version: string;
  risk_level: string;
  recommended_action: string;
}

export interface PredictionLogsResponse extends ApiResponse<{
  total: number;
  limit: number;
  offset: number;
  logs: PredictionLogItem[];
}> {}

// ─── MLflow Dashboard Types ──────────────────────────────────────────────────
export interface MLflowRun {
  run_id: string;
  run_name: string;
  experiment_id?: string;
  status: string;
  training_time: string;
  start_time: string;
  metrics: Record<string, number>;
  params: Record<string, string>;
  artifacts: string[];
}

export interface MLflowModel {
  name: string;
  current_version: string;
  stage: string;
  latest_run_id: string;
  versions_count: number;
}

export interface MLflowDashboardData {
  connected: boolean;
  experiment_name: string;
  tracking_uri: string;
  registered_models: MLflowModel[];
  active_model: string;
  current_version: string;
  latest_run: MLflowRun;
  runs: MLflowRun[];
  experiments: Array<{ experiment_id: string; name: string; lifecycle_stage: string }>;
}

export interface MLflowResponse extends ApiResponse<MLflowDashboardData> {}

// ─── History Types ───────────────────────────────────────────────────────────
export interface HistoryEntry {
  timestamp: string;
  machine_id: number;
  volt: number;
  rotate: number;
  pressure: number;
  vibration: number;
  prediction: number;
  probability: number;
  risk_level: string;
}

// ─── Live ML Experiment Types ───────────────────────────────────────────────
export interface ExperimentStep {
  id: string;
  name: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  duration_s: number;
  details: string;
}

export interface ModelComparisonResult {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  training_time_s: number;
  inference_latency_ms: number;
  is_champion: boolean;
}

export interface ExperimentStatusData {
  status: 'IDLE' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  progress: number;
  current_step: string;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  steps: ExperimentStep[];
  results: ModelComparisonResult[];
  champion: {
    model_name: string;
    f1_score: number;
    roc_auc: number;
    reason: string;
    algorithm: string;
    optimal_threshold: number;
  } | null;
  mlflow_run_id: string | null;
}

export interface ExperimentHistoryItem {
  run_id: string;
  timestamp: string;
  dataset: string;
  best_model: string;
  champion_f1: number;
  champion_roc_auc: number;
  status: string;
  models_count: number;
  models: ModelComparisonResult[];
}

// ─── Navigation ──────────────────────────────────────────────────────────────
export interface NavItem {
  label: string;
  path: string;
  icon: string;
}
