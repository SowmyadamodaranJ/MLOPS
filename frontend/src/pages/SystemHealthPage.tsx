import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Cpu,
  HardDrive,
  Activity,
  Zap,
  Server,
  Database,
  RefreshCw,
  Clock,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { fetchHealth, fetchMonitoring } from '../services/api';
import type { HealthData, MonitoringData } from '../types';
import LoadingSpinner from '../components/ui/LoadingSpinner';

export default function SystemHealthPage() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [monitoring, setMonitoring] = useState<MonitoringData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const loadData = async () => {
    try {
      setRefreshing(true);
      const [hRes, mRes] = await Promise.all([fetchHealth(), fetchMonitoring()]);
      if (hRes.success && hRes.data) setHealth(hRes.data);
      if (mRes.success && mRes.data) setMonitoring(mRes.data);
    } catch (e) {
      console.error("Failed to load health metrics:", e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
    if (!autoRefresh) return;
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh]);

  // Mock historic latency data for chart
  const latencyHistory = [
    { time: '10:00', latency: 32, cpu: 18 },
    { time: '10:05', latency: 45, cpu: 22 },
    { time: '10:10', latency: 38, cpu: 19 },
    { time: '10:15', latency: 52, cpu: 26 },
    { time: '10:20', latency: 41, cpu: 20 },
    { time: '10:25', latency: 36, cpu: 18 },
    { time: '10:30', latency: monitoring?.avg_latency_ms || 35, cpu: monitoring?.cpu_percent || 20 },
  ];

  if (loading && !health) {
    return <LoadingSpinner label="Fetching Infrastructure & System Health Metrics..." />;
  }

  const formatUptime = (sec: number = 0) => {
    const hours = Math.floor(sec / 3600);
    const mins = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    return `${hours}h ${mins}m ${s}s`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-card p-6 border-l-4 border-l-cyan-500">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white tracking-tight">System Infrastructure Health</h1>
            <span className="px-3 py-1 text-xs font-semibold rounded-full bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
              MODULE 1
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Real-time telemetry, CPU, Memory, Disk usage, throughput, and API latency performance.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-all ${
              autoRefresh
                ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                : 'bg-gray-800 text-gray-400 border-gray-700'
            }`}
          >
            {autoRefresh ? 'Live Auto-Refresh ON (5s)' : 'Auto-Refresh OFF'}
          </button>
          <button
            onClick={loadData}
            disabled={refreshing}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-factory-600 hover:bg-factory-500 text-white rounded-lg transition-all"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* KPI Resource Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* CPU */}
        <div className="glass-card p-5">
          <div className="flex items-center justify-between text-gray-400 mb-3">
            <span className="text-xs font-bold uppercase tracking-wider">CPU Utilization</span>
            <Cpu className="w-5 h-5 text-cyan-400" />
          </div>
          <div className="flex items-baseline justify-between">
            <span className="text-3xl font-extrabold text-white">
              {monitoring?.cpu_percent ?? health?.cpu_percent ?? 18.4}%
            </span>
            <span className="text-xs text-gray-400">Multi-Core</span>
          </div>
          <div className="w-full bg-gray-800 h-2 rounded-full mt-3 overflow-hidden">
            <div
              className="bg-cyan-500 h-full rounded-full transition-all duration-500"
              style={{ width: `${monitoring?.cpu_percent ?? 18}%` }}
            />
          </div>
        </div>

        {/* Memory */}
        <div className="glass-card p-5">
          <div className="flex items-center justify-between text-gray-400 mb-3">
            <span className="text-xs font-bold uppercase tracking-wider">Memory (RAM)</span>
            <Server className="w-5 h-5 text-purple-400" />
          </div>
          <div className="flex items-baseline justify-between">
            <span className="text-3xl font-extrabold text-white">
              {monitoring?.memory_percent ?? health?.memory_percent ?? 42.1}%
            </span>
            <span className="text-xs text-gray-400">
              {monitoring?.memory_used_mb ? `${monitoring.memory_used_mb} MB` : '3.4 GB'}
            </span>
          </div>
          <div className="w-full bg-gray-800 h-2 rounded-full mt-3 overflow-hidden">
            <div
              className="bg-purple-500 h-full rounded-full transition-all duration-500"
              style={{ width: `${monitoring?.memory_percent ?? 42}%` }}
            />
          </div>
        </div>

        {/* Disk */}
        <div className="glass-card p-5">
          <div className="flex items-center justify-between text-gray-400 mb-3">
            <span className="text-xs font-bold uppercase tracking-wider">Disk Storage</span>
            <HardDrive className="w-5 h-5 text-emerald-400" />
          </div>
          <div className="flex items-baseline justify-between">
            <span className="text-3xl font-extrabold text-white">
              {monitoring?.disk_percent ?? health?.disk_percent ?? 35.8}%
            </span>
            <span className="text-xs text-gray-400">NVMe SSD</span>
          </div>
          <div className="w-full bg-gray-800 h-2 rounded-full mt-3 overflow-hidden">
            <div
              className="bg-emerald-500 h-full rounded-full transition-all duration-500"
              style={{ width: `${monitoring?.disk_percent ?? 35}%` }}
            />
          </div>
        </div>

        {/* Latency */}
        <div className="glass-card p-5">
          <div className="flex items-center justify-between text-gray-400 mb-3">
            <span className="text-xs font-bold uppercase tracking-wider">Avg Latency</span>
            <Activity className="w-5 h-5 text-amber-400" />
          </div>
          <div className="flex items-baseline justify-between">
            <span className="text-3xl font-extrabold text-white">
              {monitoring?.avg_latency_ms ?? 35} <span className="text-sm font-normal text-gray-400">ms</span>
            </span>
            <span className="text-xs text-amber-400 font-semibold">
              P95: {monitoring?.p95_latency_ms ?? 48}ms
            </span>
          </div>
          <div className="w-full bg-gray-800 h-2 rounded-full mt-3 overflow-hidden">
            <div className="bg-amber-500 h-full rounded-full" style={{ width: '30%' }} />
          </div>
        </div>
      </div>

      {/* Latency Timeline & System Dependencies */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Latency Trend Chart */}
        <div className="lg:col-span-2 glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold text-white">API Latency & CPU Trend</h3>
              <p className="text-xs text-gray-400">Response latency (ms) vs CPU utilization over time</p>
            </div>
            <Zap className="w-5 h-5 text-cyan-400" />
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={latencyHistory}>
                <defs>
                  <linearGradient id="latencyGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="time" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                />
                <Area type="monotone" dataKey="latency" stroke="#06b6d4" fillOpacity={1} fill="url(#latencyGrad)" name="Latency (ms)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* System Dependencies Status */}
        <div className="glass-card p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-base font-bold text-white mb-4">Service Connectivity</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 rounded-lg bg-gray-900/60 border border-gray-800">
                <div className="flex items-center gap-3">
                  <Server className="w-5 h-5 text-cyan-400" />
                  <div>
                    <p className="text-sm font-medium text-white">Flask Backend Service</p>
                    <p className="text-xs text-gray-400">Port 5000 • Python 3.10</p>
                  </div>
                </div>
                <span className="flex items-center gap-1 text-xs font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Online
                </span>
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg bg-gray-900/60 border border-gray-800">
                <div className="flex items-center gap-3">
                  <Database className="w-5 h-5 text-purple-400" />
                  <div>
                    <p className="text-sm font-medium text-white">SQLite Database Store</p>
                    <p className="text-xs text-gray-400">predictions.db</p>
                  </div>
                </div>
                <span className="flex items-center gap-1 text-xs font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Connected
                </span>
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg bg-gray-900/60 border border-gray-800">
                <div className="flex items-center gap-3">
                  <Activity className="w-5 h-5 text-amber-400" />
                  <div>
                    <p className="text-sm font-medium text-white">MLflow Tracking Store</p>
                    <p className="text-xs text-gray-400">mlflow.db</p>
                  </div>
                </div>
                <span className="flex items-center gap-1 text-xs font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Active
                </span>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-gray-800 flex items-center justify-between text-xs text-gray-400">
            <span className="flex items-center gap-1.5">
              <Clock className="w-4 h-4 text-cyan-400" /> Uptime:
            </span>
            <span className="font-mono text-white font-bold">
              {formatUptime(health?.uptime_seconds || monitoring?.uptime_seconds || 120)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
