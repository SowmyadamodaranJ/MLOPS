import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  BrainCircuit,
  BarChart2,
  TrendingUp,
  AlertOctagon,
  CheckCircle2,
  PieChart as PieIcon,
  Sliders,
  RefreshCw,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  CartesianGrid,
} from 'recharts';
import { fetchModelMonitoring } from '../services/api';
import LoadingSpinner from '../components/ui/LoadingSpinner';

export default function ModelMonitoringPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadMetrics = async () => {
    try {
      setRefreshing(true);
      const res = await fetchModelMonitoring();
      if (res.success && res.data) {
        setData(res.data);
      }
    } catch (e) {
      console.error("Failed to load model monitoring metrics:", e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadMetrics();
  }, []);

  if (loading && !data) {
    return <LoadingSpinner label="Fetching Model Monitoring & Prediction Distributions..." />;
  }

  const metrics = data?.metrics || {};
  const modelInfo = data?.model_info || {};

  const predDistData = [
    { name: 'Normal Operation', value: metrics.prediction_distribution?.normal || 95, color: '#10b981' },
    { name: 'Predicted Failure', value: metrics.prediction_distribution?.failure || 5, color: '#ef4444' },
  ];

  const confDistData = [
    { range: 'High (>= 85%)', count: metrics.confidence_distribution?.high || 85, fill: '#3b82f6' },
    { range: 'Medium (65-84%)', count: metrics.confidence_distribution?.medium || 12, fill: '#f59e0b' },
    { range: 'Low (< 65%)', count: metrics.confidence_distribution?.low || 3, fill: '#ef4444' },
  ];

  const featStats = metrics.feature_statistics || {
    volt: { mean: 170.8, std: 14.2, min: 120.0, max: 220.0 },
    rotate: { mean: 448.2, std: 52.1, min: 250.0, max: 600.0 },
    pressure: { mean: 100.4, std: 11.5, min: 60.0, max: 150.0 },
    vibration: { mean: 40.1, std: 5.8, min: 20.0, max: 75.0 },
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-card p-6 border-l-4 border-l-purple-500">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white tracking-tight">Model Performance Monitoring</h1>
            <span className="px-3 py-1 text-xs font-semibold rounded-full bg-purple-500/20 text-purple-400 border border-purple-500/30">
              MODULE 2
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Real-time inference distribution, confidence scoring, telemetry statistics, and anomaly detection.
          </p>
        </div>

        <button
          onClick={loadMetrics}
          disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-factory-600 hover:bg-factory-500 text-white rounded-lg transition-all"
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh Metrics
        </button>
      </div>

      {/* Abnormal Behavior Banner */}
      {metrics.abnormal_behavior_detected ? (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 flex items-start gap-3">
          <AlertOctagon className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-bold text-red-200 text-sm">Abnormal Model Behavior Detected</h4>
            <ul className="list-disc list-inside text-xs mt-1 space-y-1">
              {metrics.abnormal_reasons?.map((reason: string, idx: number) => (
                <li key={idx}>{reason}</li>
              ))}
            </ul>
          </div>
        </div>
      ) : (
        <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 flex items-center gap-3">
          <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          <span className="text-xs font-semibold">Model Behavior Nominal: Predictions and confidence scores remain within control bounds.</span>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card p-5">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Failure Rate</p>
          <p className="text-3xl font-extrabold text-white mt-1">
            {((metrics.failure_prediction_rate || 0.05) * 100).toFixed(1)}%
          </p>
          <p className="text-xs text-gray-500 mt-1">Expected range: 2% - 10%</p>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Avg Model Confidence</p>
          <p className="text-3xl font-extrabold text-emerald-400 mt-1">
            {((metrics.avg_confidence || 0.942) * 100).toFixed(1)}%
          </p>
          <p className="text-xs text-gray-500 mt-1">Target threshold: &gt; 85%</p>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Samples Evaluated</p>
          <p className="text-3xl font-extrabold text-white mt-1">
            {metrics.sample_count ?? 120}
          </p>
          <p className="text-xs text-gray-500 mt-1">Sliding inference window</p>
        </div>

        <div className="glass-card p-5">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Active Algorithm</p>
          <p className="text-xl font-bold text-purple-400 mt-1 truncate">
            {modelInfo.algorithm || 'XGBoost'}
          </p>
          <p className="text-xs text-gray-500 mt-1">Version: {modelInfo.version || 'v2.0.0'}</p>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Prediction Distribution */}
        <div className="glass-card p-6">
          <h3 className="text-base font-bold text-white mb-2">Prediction Distribution</h3>
          <p className="text-xs text-gray-400 mb-4">Ratio of Healthy vs Machine Failure Predictions</p>
          <div className="h-64 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={predDistData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {predDistData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confidence Score Distribution */}
        <div className="glass-card p-6">
          <h3 className="text-base font-bold text-white mb-2">Confidence Score Distribution</h3>
          <p className="text-xs text-gray-400 mb-4">Model certainty grouping across inference requests</p>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={confDistData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="range" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Input Feature Telemetry Statistics */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-white">Input Feature Telemetry Statistics</h3>
            <p className="text-xs text-gray-400">Live mean, std, min, and max for incoming sensor inputs</p>
          </div>
          <Sliders className="w-5 h-5 text-purple-400" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Object.entries(featStats).map(([feat, stats]: [string, any]) => (
            <div key={feat} className="p-4 rounded-xl bg-gray-900/60 border border-gray-800 space-y-2">
              <span className="text-xs font-bold text-purple-400 uppercase">{feat}</span>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <p className="text-gray-400">Mean</p>
                  <p className="font-bold text-white">{stats.mean}</p>
                </div>
                <div>
                  <p className="text-gray-400">Std Dev</p>
                  <p className="font-bold text-white">{stats.std}</p>
                </div>
                <div>
                  <p className="text-gray-400">Min</p>
                  <p className="font-bold text-emerald-400">{stats.min}</p>
                </div>
                <div>
                  <p className="text-gray-400">Max</p>
                  <p className="font-bold text-amber-400">{stats.max}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
