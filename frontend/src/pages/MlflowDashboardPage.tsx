import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
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
} from 'lucide-react';
import { fetchMLflowDashboard } from '../services/api';
import type { MLflowDashboardData } from '../types';
import LoadingSpinner from '../components/ui/LoadingSpinner';

export default function MlflowDashboardPage() {
  const [data, setData] = useState<MLflowDashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadMLflow = async () => {
    try {
      setRefreshing(true);
      const res = await fetchMLflowDashboard();
      if (res.success && res.data) {
        setData(res.data);
      }
    } catch (e) {
      console.error("Failed to load MLflow telemetry:", e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadMLflow();
  }, []);

  if (loading && !data) {
    return <LoadingSpinner label="Connecting directly to MLflow Experiment Tracking Store..." />;
  }

  const latestRun = data?.latest_run;
  const regModels = data?.registered_models || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-card p-6 border-l-4 border-l-amber-500">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white tracking-tight">MLflow Experiment & Registry Dashboard</h1>
            <span className="px-3 py-1 text-xs font-semibold rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30">
              MODULE 8
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Native MLflow integration reading experiment runs, hyperparameter logs, registered models, and artifacts.
          </p>
        </div>

        <button
          onClick={loadMLflow}
          disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-factory-600 hover:bg-factory-500 text-white rounded-lg transition-all"
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh MLflow
        </button>
      </div>

      {/* Primary Status Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card p-5">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Experiment Name</p>
          <p className="text-lg font-bold text-white mt-1 truncate">{data?.experiment_name}</p>
          <p className="text-xs text-amber-400 mt-1">Tracking URI: {data?.tracking_uri}</p>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Registered Model</p>
          <p className="text-lg font-bold text-cyan-400 mt-1 truncate">
            {regModels[0]?.name || 'SmartFactory_XGBoost'}
          </p>
          <p className="text-xs text-gray-400 mt-1">Stage: {regModels[0]?.stage || 'Production'}</p>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Active Version</p>
          <p className="text-3xl font-extrabold text-emerald-400 mt-1">
            {data?.current_version || 'v2.0.0'}
          </p>
          <p className="text-xs text-gray-400 mt-1">Total Versions: {regModels[0]?.versions_count || 2}</p>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Latest Training Time</p>
          <p className="text-2xl font-bold text-white mt-1">
            {latestRun?.training_time || '14.2s'}
          </p>
          <p className="text-xs text-gray-400 mt-1">{latestRun?.start_time || '2026-07-23'}</p>
        </div>
      </div>

      {/* Latest Run Deep Dive */}
      {latestRun && (
        <div className="glass-card p-6 border-t-2 border-t-cyan-500">
          <div className="flex items-center justify-between mb-4">
            <div>
              <span className="text-xs font-bold text-cyan-400 uppercase">ACTIVE PRODUCTION RUN</span>
              <h3 className="text-lg font-bold text-white">{latestRun.run_name}</h3>
              <p className="text-xs font-mono text-gray-400">Run ID: {latestRun.run_id}</p>
            </div>
            <span className="px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 font-bold text-xs border border-emerald-500/30 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4" /> {latestRun.status}
            </span>
          </div>

          {/* Metrics & Params Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
            {/* Metrics */}
            <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800">
              <h4 className="text-sm font-bold text-white mb-3">Logged Evaluation Metrics</h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                {Object.entries(latestRun.metrics || {}).map(([key, val]) => (
                  <div key={key} className="p-2.5 rounded-lg bg-gray-950/60 border border-gray-850">
                    <span className="text-gray-400 uppercase text-[10px] font-bold">{key}</span>
                    <p className="text-base font-extrabold text-cyan-400 mt-0.5">{typeof val === 'number' ? val.toFixed(4) : val}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Hyperparameters */}
            <div className="p-4 rounded-xl bg-gray-900/60 border border-gray-800">
              <h4 className="text-sm font-bold text-white mb-3">Hyperparameter Configuration</h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                {Object.entries(latestRun.params || {}).map(([key, val]) => (
                  <div key={key} className="p-2.5 rounded-lg bg-gray-950/60 border border-gray-850">
                    <span className="text-gray-400 uppercase text-[10px] font-bold">{key}</span>
                    <p className="text-sm font-mono text-white mt-0.5">{val}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Artifacts */}
          <div className="mt-6 pt-4 border-t border-gray-800">
            <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
              <Box className="w-4 h-4 text-amber-400" /> Logged MLflow Artifacts
            </h4>
            <div className="flex flex-wrap gap-2 text-xs">
              {latestRun.artifacts?.map((art, idx) => (
                <span key={idx} className="px-3 py-1.5 rounded-lg bg-gray-900 border border-gray-800 text-gray-300 font-mono flex items-center gap-1.5">
                  <FileCode className="w-3.5 h-3.5 text-cyan-400" /> {art}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
