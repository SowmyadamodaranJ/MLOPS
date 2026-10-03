import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  GitBranch,
  Box,
  Layers,
  Clock,
  CheckCircle2,
  FileCode,
  Sparkles,
  ExternalLink,
  RefreshCw,
  Play,
  Award,
  AlertCircle,
  Zap,
  Check,
  ChevronRight,
  TrendingUp,
  Cpu,
  BarChart2,
  Database,
  History as HistoryIcon,
} from 'lucide-react';
import {
  fetchMLflowDashboard,
  runExperiment,
  fetchExperimentStatus,
  fetchExperimentHistory,
  postPromoteChampion,
} from '../services/api';
import type {
  MLflowDashboardData,
  ExperimentStatusData,
  ModelComparisonResult,
  ExperimentHistoryItem,
} from '../types';
import LoadingSpinner from '../components/ui/LoadingSpinner';

export default function MlflowDashboardPage() {
  const [data, setData] = useState<MLflowDashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Experiment state
  const [expStatus, setExpStatus] = useState<ExperimentStatusData | null>(null);
  const [history, setHistory] = useState<ExperimentHistoryItem[]>([]);
  const [isStarting, setIsStarting] = useState(false);
  const [promoteMsg, setPromoteMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [selectedHistoryRun, setSelectedHistoryRun] = useState<ExperimentHistoryItem | null>(null);

  const pollingRef = useRef<number | null>(null);

  const loadMLflow = async () => {
    try {
      setRefreshing(true);
      const res = await fetchMLflowDashboard();
      if (res.success && res.data) {
        setData(res.data);
      }
    } catch (e) {
      console.error('Failed to load MLflow telemetry:', e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const loadExperimentStatus = async () => {
    try {
      const res = await fetchExperimentStatus();
      if (res.success && res.data) {
        setExpStatus(res.data);
        return res.data;
      }
    } catch (e) {
      console.error('Failed to load experiment status:', e);
    }
    return null;
  };

  const loadHistory = async () => {
    try {
      const res = await fetchExperimentHistory();
      if (res.success && res.data) {
        setHistory(res.data);
      }
    } catch (e) {
      console.error('Failed to load experiment history:', e);
    }
  };

  useEffect(() => {
    loadMLflow();
    loadExperimentStatus();
    loadHistory();

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, []);

  // Poll experiment status when running
  useEffect(() => {
    if (expStatus?.status === 'RUNNING') {
      if (!pollingRef.current) {
        pollingRef.current = window.setInterval(async () => {
          const current = await loadExperimentStatus();
          if (current && current.status !== 'RUNNING') {
            if (pollingRef.current) {
              clearInterval(pollingRef.current);
              pollingRef.current = null;
            }
            loadMLflow();
            loadHistory();
          }
        }, 800);
      }
    } else {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }
  }, [expStatus?.status]);

  const handleStartExperiment = async () => {
    try {
      setIsStarting(true);
      setPromoteMsg(null);
      const res = await runExperiment();
      if (res.success) {
        await loadExperimentStatus();
      }
    } catch (e) {
      console.error('Failed to run experiment:', e);
    } finally {
      setIsStarting(false);
    }
  };

  const handlePromoteChampion = async (modelName: string) => {
    try {
      const res = await postPromoteChampion(modelName);
      if (res.success) {
        setPromoteMsg({
          type: 'success',
          text: `Successfully promoted ${modelName} to Production Stage in MLflow Model Registry!`,
        });
        loadMLflow();
      } else {
        setPromoteMsg({
          type: 'error',
          text: res.message || 'Failed to promote model.',
        });
      }
    } catch (e) {
      setPromoteMsg({
        type: 'error',
        text: 'Network error promoting champion candidate.',
      });
    }
  };

  if (loading && !data) {
    return <LoadingSpinner label="Connecting directly to MLflow Experiment Tracking Store..." />;
  }

  const latestRun = data?.latest_run;
  const regModels = data?.registered_models || [];

  // Decide which comparison table to display: selected history or current/last experiment
  const displayResults: ModelComparisonResult[] =
    selectedHistoryRun?.models || expStatus?.results || [];

  const displayChampion =
    selectedHistoryRun
      ? {
          model_name: selectedHistoryRun.best_model,
          f1_score: selectedHistoryRun.champion_f1,
          roc_auc: selectedHistoryRun.champion_roc_auc,
          reason: 'Highest validated F1-score in historical run',
        }
      : expStatus?.champion;

  return (
    <div className="space-y-6">
      {/* Header & Main Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 glass-card p-6 border-l-4 border-l-amber-500">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white tracking-tight">
              MLflow Experiment & Model Comparison
            </h1>
            <span className="px-3 py-1 text-xs font-semibold rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30">
              MODULE 8
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Execute live 4-model training (LR, DT, RF, XGBoost) on authentic telemetry data with strict chronological validation.
          </p>
        </div>

        {/* Experiment Action Buttons */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            id="btn-run-experiment"
            onClick={handleStartExperiment}
            disabled={isStarting || expStatus?.status === 'RUNNING'}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm font-semibold rounded-lg shadow-lg transition-all ${
              expStatus?.status === 'RUNNING'
                ? 'bg-amber-600/60 text-white cursor-not-allowed animate-pulse'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-950/40'
            }`}
          >
            {expStatus?.status === 'RUNNING' ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-amber-200" />
                Training 4 Models...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Run 4-Model Experiment
              </>
            )}
          </button>

          <button
            onClick={() => {
              loadMLflow();
              loadExperimentStatus();
              loadHistory();
            }}
            disabled={refreshing}
            className="flex items-center gap-2 px-3.5 py-2 text-sm font-medium bg-gray-800 hover:bg-gray-700 text-gray-200 rounded-lg transition-all border border-gray-700"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>

          <a
            href={data?.tracking_uri?.startsWith('http') ? data.tracking_uri : 'http://localhost:5001'}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium bg-cyan-950/60 hover:bg-cyan-900/80 text-cyan-300 rounded-lg transition-all border border-cyan-800/60"
          >
            <ExternalLink className="w-4 h-4" />
            MLflow UI (5001)
          </a>
        </div>
      </div>

      {/* Promotion Feedback Notification */}
      <AnimatePresence>
        {promoteMsg && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className={`p-4 rounded-xl flex items-center justify-between border ${
              promoteMsg.type === 'success'
                ? 'bg-emerald-950/50 border-emerald-700/60 text-emerald-200'
                : 'bg-rose-950/50 border-rose-700/60 text-rose-200'
            }`}
          >
            <div className="flex items-center gap-2 text-sm font-medium">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <span>{promoteMsg.text}</span>
            </div>
            <button
              onClick={() => setPromoteMsg(null)}
              className="text-xs text-gray-400 hover:text-white px-2 py-1"
            >
              Dismiss
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Top Metadata Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card p-5">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-cyan-400" /> Experiment Name
          </p>
          <p className="text-lg font-bold text-white mt-1 truncate">{data?.experiment_name}</p>
          <p className="text-xs text-amber-400 mt-1 truncate">URI: {data?.tracking_uri}</p>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
            <Award className="w-3.5 h-3.5 text-amber-400" /> Production Champion
          </p>
          <p className="text-lg font-bold text-cyan-400 mt-1 truncate">
            {regModels[0]?.name || 'SmartFactory_XGBoost'}
          </p>
          <p className="text-xs text-gray-400 mt-1">Stage: {regModels[0]?.stage || 'Production'}</p>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
            <GitBranch className="w-3.5 h-3.5 text-emerald-400" /> Active Version
          </p>
          <p className="text-3xl font-extrabold text-emerald-400 mt-1">
            {data?.current_version || 'v2.0.0'}
          </p>
          <p className="text-xs text-gray-400 mt-1">Threshold: 0.8781</p>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-indigo-400" /> Latency & Training
          </p>
          <p className="text-2xl font-bold text-white mt-1">
            ~12.5 ms
          </p>
          <p className="text-xs text-gray-400 mt-1">Last trained: {latestRun?.start_time || 'Recent'}</p>
        </div>
      </div>

      {/* Live ML Experiment Progress Section */}
      <div className="glass-card p-6 border-t-2 border-t-amber-500">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-amber-400 uppercase tracking-wide flex items-center gap-1.5">
                <Cpu className="w-4 h-4" /> ML Experiment Pipeline Execution
              </span>
              <span
                className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${
                  expStatus?.status === 'RUNNING'
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse'
                    : expStatus?.status === 'COMPLETED'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    : 'bg-gray-800 text-gray-300 border border-gray-700'
                }`}
              >
                {expStatus?.status || 'IDLE'}
              </span>
            </div>
            <h3 className="text-lg font-bold text-white mt-1">
              Live Sequential 4-Model Training & Cross-Evaluation
            </h3>
          </div>

          <div className="text-right">
            <span className="text-xs text-gray-400">Pipeline Progress</span>
            <div className="text-xl font-extrabold text-cyan-400">
              {expStatus?.progress || 0}%
            </div>
          </div>
        </div>

        {/* Real-time Progress Bar */}
        <div className="w-full bg-gray-900 rounded-full h-2.5 mb-6 overflow-hidden border border-gray-800">
          <div
            className={`h-2.5 rounded-full transition-all duration-300 ${
              expStatus?.status === 'RUNNING'
                ? 'bg-gradient-to-r from-amber-500 via-cyan-500 to-emerald-500'
                : expStatus?.status === 'COMPLETED'
                ? 'bg-emerald-500'
                : 'bg-gray-700'
            }`}
            style={{ width: `${expStatus?.progress || 0}%` }}
          />
        </div>

        {/* Step-by-Step Live Stepper */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {(expStatus?.steps || []).map((step, idx) => {
            const isCompleted = step.status === 'COMPLETED';
            const isRunning = step.status === 'RUNNING';
            const isFailed = step.status === 'FAILED';

            return (
              <div
                key={step.id || idx}
                className={`p-3.5 rounded-xl border transition-all ${
                  isRunning
                    ? 'bg-amber-950/30 border-amber-500/60 shadow-lg shadow-amber-950/20'
                    : isCompleted
                    ? 'bg-gray-900/60 border-emerald-500/40'
                    : isFailed
                    ? 'bg-rose-950/30 border-rose-500/50'
                    : 'bg-gray-950/40 border-gray-850 opacity-60'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[11px] font-bold text-gray-400">
                    STEP {idx + 1}
                  </span>
                  {isRunning && (
                    <RefreshCw className="w-4 h-4 text-amber-400 animate-spin" />
                  )}
                  {isCompleted && (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  )}
                  {isFailed && (
                    <AlertCircle className="w-4 h-4 text-rose-400" />
                  )}
                  {!isRunning && !isCompleted && !isFailed && (
                    <div className="w-3.5 h-3.5 rounded-full border border-gray-600" />
                  )}
                </div>
                <h4 className="text-xs font-bold text-white truncate">{step.name}</h4>
                <p className="text-[11px] text-gray-400 mt-1 line-clamp-2 leading-relaxed">
                  {step.details || 'Pending execution...'}
                </p>
                {step.duration_s > 0 && (
                  <p className="text-[10px] text-cyan-400 font-mono mt-2">
                    Duration: {step.duration_s.toFixed(2)}s
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Model Comparison Table Section */}
      <div className="glass-card p-6 border-t-2 border-t-cyan-500">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-cyan-400 uppercase tracking-wide flex items-center gap-1.5">
                <BarChart2 className="w-4 h-4" /> Live Experiment Benchmark
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                Newly Executed 4-Model Workflow
              </span>
              {selectedHistoryRun && (
                <span className="text-xs text-amber-400 px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30">
                  Viewing Historical Run: {selectedHistoryRun.run_id.slice(0, 16)}...
                </span>
              )}
            </div>
            <h3 className="text-lg font-bold text-white mt-1">
              Live Benchmark on 31-Feature Chronological Test Partition
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">
              All 4 models evaluated sequentially on the authentic single-year chronological holdout partition (70/30 cutoff). Primary selection metric: <span className="text-emerald-400 font-semibold">F1-Score</span>.
            </p>
          </div>

          {selectedHistoryRun && (
            <button
              onClick={() => setSelectedHistoryRun(null)}
              className="text-xs text-cyan-400 hover:text-cyan-300 underline font-medium self-start sm:self-auto"
            >
              Reset to Current Run
            </button>
          )}
        </div>

        {/* Methodology Reconciliation Banner */}
        <div className="mb-4 p-3 rounded-lg bg-blue-950/30 border border-blue-800/40 text-xs text-gray-300 flex items-start gap-2">
          <Sparkles className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-cyan-300">Methodology Note:</span> Live Experiment results are generated dynamically on the single-year 70/30 holdout split. The persisted Production Champion baseline (F1: 0.8250, ROC-AUC: 0.8870) represents the 5-fold TimeSeries cross-validated benchmark documented in the model manifest.
          </div>
        </div>

        {/* Results Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-gray-300">
            <thead className="bg-gray-900/80 text-xs uppercase font-bold text-gray-400 border-b border-gray-800">
              <tr>
                <th className="py-3 px-4">Model</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Precision</th>
                <th className="py-3 px-4">Recall</th>
                <th className="py-3 px-4 text-emerald-400">F1-Score (Primary)</th>
                <th className="py-3 px-4">ROC-AUC</th>
                <th className="py-3 px-4">Train Time (s)</th>
                <th className="py-3 px-4">Latency (ms)</th>
                <th className="py-3 px-4 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/60 font-mono text-xs">
              {displayResults.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-8 text-center text-gray-500 font-sans">
                    No experiment executed in current session yet. Click <span className="text-emerald-400 font-semibold">"Run 4-Model Experiment"</span> above to train and compare all 4 models live.
                  </td>
                </tr>
              ) : (
                displayResults.map((r, i) => (
                  <tr
                    key={r.model || i}
                    className={`hover:bg-gray-800/40 transition-colors ${
                      r.is_champion ? 'bg-emerald-950/20 border-l-4 border-l-emerald-500' : ''
                    }`}
                  >
                    <td className="py-3.5 px-4 font-sans font-bold text-white flex items-center gap-2">
                      {r.is_champion && (
                        <Award className="w-4 h-4 text-amber-400 fill-amber-400/20" />
                      )}
                      <span>{r.model}</span>
                    </td>
                    <td className="py-3.5 px-4 text-gray-200">{r.accuracy.toFixed(4)}</td>
                    <td className="py-3.5 px-4 text-gray-200">{r.precision.toFixed(4)}</td>
                    <td className="py-3.5 px-4 text-gray-200">{r.recall.toFixed(4)}</td>
                    <td className="py-3.5 px-4 font-bold text-emerald-400 text-sm">
                      {r.f1_score.toFixed(4)}
                    </td>
                    <td className="py-3.5 px-4 text-cyan-300">{r.roc_auc.toFixed(4)}</td>
                    <td className="py-3.5 px-4 text-gray-400">{r.training_time_s.toFixed(2)}s</td>
                    <td className="py-3.5 px-4 text-gray-400">{r.inference_latency_ms.toFixed(1)} ms</td>
                    <td className="py-3.5 px-4 text-right font-sans">
                      {r.is_champion ? (
                        <span className="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 text-[11px] font-bold border border-emerald-500/40">
                          Champion
                        </span>
                      ) : (
                        <span className="text-gray-500 text-[11px]">Evaluated</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Champion Decision Card */}
        {displayChampion && (
          <div className="mt-6 p-4 rounded-xl bg-gradient-to-r from-emerald-950/40 via-gray-900 to-cyan-950/40 border border-emerald-500/40 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-start gap-3">
              <div className="p-2.5 rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-emerald-300">
                <Award className="w-6 h-6" />
              </div>
              <div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400">
                  AUTOMATED CHAMPION SELECTION
                </span>
                <h4 className="text-base font-bold text-white">
                  Champion Model: {displayChampion.model_name}
                </h4>
                <p className="text-xs text-gray-300 mt-0.5">
                  {displayChampion.reason ||
                    `Achieved top F1-Score of ${displayChampion.f1_score.toFixed(
                      4
                    )} and ROC-AUC of ${displayChampion.roc_auc.toFixed(4)}.`}
                </p>
              </div>
            </div>

            <button
              onClick={() => handlePromoteChampion(displayChampion.model_name)}
              className="flex items-center gap-1.5 px-4 py-2 text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition-all shadow-md self-start md:self-auto"
            >
              <Check className="w-3.5 h-3.5" />
              Register as Production Champion
            </button>
          </div>
        )}
      </div>

      {/* Production Model Run Deep Dive (Authoritative MLflow Run) */}
      {latestRun && (
        <div className="glass-card p-6 border-t-2 border-t-indigo-500">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-indigo-400 uppercase tracking-wide">
                  Production Champion Evaluation
                </span>
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  XGBoost v2.0.0 (Authoritative Manifest)
                </span>
              </div>
              <h3 className="text-lg font-bold text-white mt-1">{latestRun.run_name}</h3>
              <p className="text-xs font-mono text-gray-400">MLflow Run ID: {latestRun.run_id}</p>
            </div>
            <span className="px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 font-bold text-xs border border-emerald-500/30 flex items-center gap-1.5 self-start sm:self-auto">
              <CheckCircle2 className="w-4 h-4" /> {latestRun.status}
            </span>
          </div>

          {/* Metrics & Params Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            {/* Dynamic Authoritative Metrics */}
            <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800">
              <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-400" /> Authoritative Evaluation Metrics
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                {Object.entries(latestRun.metrics || {}).map(([key, val]) => (
                  <div key={key} className="p-2.5 rounded-lg bg-gray-950/60 border border-gray-850">
                    <span className="text-gray-400 uppercase text-[10px] font-bold">{key}</span>
                    <p className="text-base font-extrabold text-cyan-400 mt-0.5">
                      {typeof val === 'number' ? val.toFixed(4) : val}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Hyperparameters */}
            <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800">
              <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
                <Layers className="w-4 h-4 text-indigo-400" /> Hyperparameter Configuration
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                {Object.entries(latestRun.params || {}).map(([key, val]) => (
                  <div key={key} className="p-2.5 rounded-lg bg-gray-950/60 border border-gray-850">
                    <span className="text-gray-400 uppercase text-[10px] font-bold">{key}</span>
                    <p className="text-sm font-mono text-white mt-0.5 truncate">{val}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Artifacts */}
          <div className="mt-6 pt-4 border-t border-gray-800">
            <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
              <Box className="w-4 h-4 text-amber-400" /> Authoritative MLflow Artifacts
            </h4>
            <div className="flex flex-wrap gap-2 text-xs">
              {latestRun.artifacts?.map((art, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1.5 rounded-lg bg-gray-900 border border-gray-800 text-gray-300 font-mono flex items-center gap-1.5"
                >
                  <FileCode className="w-3.5 h-3.5 text-cyan-400" /> {art}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Experiment History */}
      <div className="glass-card p-6 border-t-2 border-t-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="flex items-center gap-2">
              <HistoryIcon className="w-4 h-4 text-cyan-400" />
              <h3 className="text-lg font-bold text-white">Experiment History</h3>
            </div>
            <p className="text-xs text-gray-400 mt-0.5">
              Chronological log of past multi-model runs stored in MLflow and local registry.
            </p>
          </div>
          <span className="text-xs text-gray-400 font-mono">{history.length} runs recorded</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-gray-300">
            <thead className="bg-gray-900/80 text-xs uppercase font-bold text-gray-400 border-b border-gray-800">
              <tr>
                <th className="py-3 px-4">Run ID</th>
                <th className="py-3 px-4">Date & Time</th>
                <th className="py-3 px-4">Dataset Partition</th>
                <th className="py-3 px-4">Best Model</th>
                <th className="py-3 px-4">Champion F1</th>
                <th className="py-3 px-4">Champion ROC-AUC</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/60 font-mono text-xs">
              {history.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-6 text-center text-gray-500 font-sans">
                    No previous experiment history found.
                  </td>
                </tr>
              ) : (
                history.map((item, idx) => (
                  <tr
                    key={item.run_id || idx}
                    className="hover:bg-gray-800/30 transition-colors cursor-pointer"
                    onClick={() => setSelectedHistoryRun(item)}
                  >
                    <td className="py-3 px-4 text-cyan-400 font-bold truncate max-w-[120px]">
                      {item.run_id.slice(0, 16)}...
                    </td>
                    <td className="py-3 px-4 text-gray-300 font-sans">{item.timestamp}</td>
                    <td className="py-3 px-4 text-gray-400 font-sans">{item.dataset}</td>
                    <td className="py-3 px-4 text-white font-sans font-semibold">
                      {item.best_model}
                    </td>
                    <td className="py-3 px-4 text-emerald-400 font-bold">
                      {item.champion_f1 ? item.champion_f1.toFixed(4) : '-'}
                    </td>
                    <td className="py-3 px-4 text-cyan-300">
                      {item.champion_roc_auc ? item.champion_roc_auc.toFixed(4) : '-'}
                    </td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 text-[10px] font-bold border border-emerald-500/20 font-sans">
                        {item.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right font-sans">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedHistoryRun(item);
                        }}
                        className="text-xs text-cyan-400 hover:text-cyan-300 font-medium"
                      >
                        View Comparison
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
