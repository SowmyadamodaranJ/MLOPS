/**
 * api.ts
 * ------
 * Axios instance and typed service functions for the Predictive
 * Maintenance frontend platform.
 */

import axios, { AxiosError } from 'axios';
import type {
  DashboardData,
  DashboardResponse,
  MachinesResponse,
  PredictionResponse,
  ModelData,
  ModelResponse,
  MonitoringData,
  MonitoringResponse,
  HealthData,
  HealthResponse,
  AlertsResponse,
  PredictionLogsResponse,
  MLflowResponse,
  MLflowDashboardData,
} from '../types';

export class ApiError extends Error {
  errorCode: string;
  status: number;
  override message: string;

  constructor(message: string, errorCode: string, status: number) {
    super(message);
    this.name       = 'ApiError';
    this.message    = message;
    this.errorCode  = errorCode;
    this.status     = status;
  }
}

const api = axios.create({
  baseURL: '/api',
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.response.use(
  (response) => response,

  (axiosError: AxiosError<any>) => {
    if (!axiosError.response) {
      const isTimeout = axiosError.code === 'ECONNABORTED';
      throw new ApiError(
        isTimeout
          ? 'The request timed out. Please check your connection and try again.'
          : 'Cannot reach the backend server. Please ensure it is running.',
        isTimeout ? 'REQUEST_TIMEOUT' : 'NETWORK_ERROR',
        -1,
      );
    }

    const body        = axiosError.response.data || {};
    const status      = axiosError.response.status;
    const errorCode   = body.error_code || body.errorCode || 'UNKNOWN_ERROR';
    const message     = body.message    || axiosError.message || 'An unexpected error occurred.';

    console.error(`[API ${status}] [${errorCode}] ${message}`);
    throw new ApiError(message, errorCode, status);
  },
);

export const fetchDashboard = () =>
  api.get<DashboardResponse>('/dashboard').then((r) => r.data.data!);

export const fetchMachines = () =>
  api.get<MachinesResponse>('/machines').then((r) => r.data.data!);

export const fetchMachineFeatures = (machineId: number) =>
  api.get(`/machines/${machineId}/features`).then((r) => r.data.data!);

export const fetchMachineDecision = (machineId: number) =>
  api.get(`/machines/${machineId}/decision`).then((r) => r.data.data!);

export const fetchHealthQueue = (riskTier?: string, limit?: number) => {
  const params: Record<string, any> = {};
  if (riskTier) params.risk_tier = riskTier;
  if (limit) params.limit = limit;
  return api.get('/machines/health-queue', { params }).then((r) => r.data.data!);
};

export const postPrediction = (features: Record<string, any>) =>
  api.post<PredictionResponse>('/predict', features).then((r) => r.data);

export const postExplain = (features: Record<string, any>) =>
  api.post('/explain', features).then((r) => r.data);

export const fetchMetrics = () =>
  api.get('/metrics').then((r) => r.data.data!);

export const fetchModels = () =>
  api.get<ModelResponse>('/models').then((r) => r.data.data!);

export const fetchMonitoring = () =>
  api.get<MonitoringResponse>('/monitoring').then((r) => r.data);

export const fetchHealth = () =>
  api.get<HealthResponse>('/health').then((r) => r.data);

export const fetchHistory = () =>
  api.get('/history').then((r) => r.data.data!);

// ─── Module 10 Extensions ───────────────────────────────────────────────────

export const fetchModelMonitoring = () =>
  api.get('/monitoring/model').then((r) => r.data);

export const fetchDriftMonitoring = () =>
  api.get('/monitoring/drift').then((r) => r.data);

export const generateDriftReport = () =>
  api.post('/monitoring/drift/generate').then((r) => r.data);

export const fetchAlerts = () =>
  api.get<AlertsResponse>('/monitoring/alerts').then((r) => r.data);

export const postAcknowledgeAlert = (alertId: string) =>
  api.post('/monitoring/alerts/acknowledge', { alert_id: alertId }).then((r) => r.data);

export const fetchPredictionLogs = (params: {
  search?: string;
  machine_id?: number;
  risk_level?: string;
  prediction?: number;
  limit?: number;
  offset?: number;
}) =>
  api.get<PredictionLogsResponse>('/monitoring/logs', { params }).then((r) => r.data);

export const fetchMLflowDashboard = () =>
  api.get<MLflowResponse>('/monitoring/mlflow').then((r) => r.data);

export const getExportLogsUrl = (search?: string, riskLevel?: string) => {
  const query = new URLSearchParams();
  if (search) query.append('search', search);
  if (riskLevel) query.append('risk_level', riskLevel);
  return `/api/monitoring/logs/export?${query.toString()}`;
};

// ─── Live ML Experiment Workflow ─────────────────────────────────────────────

export const runExperiment = () =>
  api.post('/experiment/run').then((r) => r.data);

export const fetchExperimentStatus = () =>
  api.get('/experiment/status').then((r) => r.data);

export const fetchExperimentHistory = () =>
  api.get('/experiment/history').then((r) => r.data);

export const postPromoteChampion = (modelName: string) =>
  api.post('/experiment/promote', { model_name: modelName }).then((r) => r.data);

export default api;
