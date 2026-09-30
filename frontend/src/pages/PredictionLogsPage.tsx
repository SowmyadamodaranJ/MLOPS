import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  ScrollText,
  Search,
  Download,
  Filter,
  RefreshCw,
  SlidersHorizontal,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  AlertTriangle,
} from 'lucide-react';
import { fetchPredictionLogs, getExportLogsUrl } from '../services/api';
import type { PredictionLogItem } from '../types';
import LoadingSpinner from '../components/ui/LoadingSpinner';

export default function PredictionLogsPage() {
  const [logs, setLogs] = useState<PredictionLogItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const [searchTerm, setSearchTerm] = useState('');
  const [riskFilter, setRiskFilter] = useState('ALL');
  const [page, setPage] = useState(0);
  const limit = 15;

  const loadLogs = async () => {
    try {
      setRefreshing(true);
      const res = await fetchPredictionLogs({
        search: searchTerm || undefined,
        risk_level: riskFilter !== 'ALL' ? riskFilter : undefined,
        limit,
        offset: page * limit,
      });
      if (res.success && res.data) {
        setLogs(res.data.logs || []);
        setTotal(res.data.total || 0);
      }
    } catch (e) {
      console.error("Failed to fetch prediction logs:", e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadLogs();
  }, [searchTerm, riskFilter, page]);

  if (loading && logs.length === 0) {
    return <LoadingSpinner label="Querying Prediction Audit Logs Database..." />;
  }

  const exportUrl = getExportLogsUrl(searchTerm, riskFilter !== 'ALL' ? riskFilter : undefined);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-card p-6 border-l-4 border-l-cyan-500">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white tracking-tight">Prediction Audit Logs & Search</h1>
            <span className="px-3 py-1 text-xs font-semibold rounded-full bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
              MODULE 7
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Complete persistent audit trail storing inputs, probabilities, confidence scores, latencies, and prediction IDs.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <a
            href={exportUrl}
            download="prediction_logs.csv"
            className="flex items-center gap-2 px-4 py-2 text-sm font-semibold bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg transition-all shadow-lg shadow-emerald-600/20"
          >
            <Download className="w-4 h-4" /> Export to CSV
          </a>
          <button
            onClick={loadLogs}
            disabled={refreshing}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-factory-600 hover:bg-factory-500 text-white rounded-lg transition-all"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Search & Filter Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 glass-card p-4">
        {/* Search input */}
        <div className="relative md:col-span-2">
          <Search className="w-4 h-4 absolute left-3.5 top-3 text-gray-400" />
          <input
            type="text"
            placeholder="Search by Prediction ID, Machine ID, or Risk Level..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setPage(0);
            }}
            className="w-full pl-10 pr-4 py-2 bg-gray-900/80 border border-gray-800 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500"
          />
        </div>

        {/* Risk Level Filter */}
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-gray-400 shrink-0" />
          <select
            value={riskFilter}
            onChange={(e) => {
              setRiskFilter(e.target.value);
              setPage(0);
            }}
            className="w-full py-2 px-3 bg-gray-900/80 border border-gray-800 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-500"
          >
            <option value="ALL">All Risk Levels</option>
            <option value="High">High Risk Only</option>
            <option value="Warning">Warning Level</option>
            <option value="Low">Low Risk</option>
          </select>
        </div>
      </div>

      {/* Audit Log Table */}
      <div className="glass-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-gray-800 bg-gray-900/50 text-gray-400 font-bold uppercase tracking-wider">
                <th className="py-3.5 px-4">Prediction ID</th>
                <th className="py-3.5 px-4">Timestamp</th>
                <th className="py-3.5 px-4">Machine ID</th>
                <th className="py-3.5 px-4">Telemetry Inputs (V / R / P / Vib)</th>
                <th className="py-3.5 px-4">Prediction</th>
                <th className="py-3.5 px-4">Probability</th>
                <th className="py-3.5 px-4">Confidence</th>
                <th className="py-3.5 px-4">Latency</th>
                <th className="py-3.5 px-4">Version</th>
                <th className="py-3.5 px-4">Risk Level</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/60 text-gray-300">
              {logs.length === 0 ? (
                <tr>
                  <td colSpan={10} className="py-8 text-center text-gray-500">
                    No prediction logs match the search filters.
                  </td>
                </tr>
              ) : (
                logs.map((log) => (
                  <tr key={log.prediction_id || log.id} className="hover:bg-white/[0.02]">
                    <td className="py-3.5 px-4 font-mono font-bold text-cyan-400">{log.prediction_id}</td>
                    <td className="py-3.5 px-4 text-gray-400 font-mono">{log.timestamp}</td>
                    <td className="py-3.5 px-4 font-bold text-white"># {log.machine_id}</td>
                    <td className="py-3.5 px-4 font-mono text-gray-300">
                      {log.volt.toFixed(1)}v | {log.rotate.toFixed(0)}rpm | {log.pressure.toFixed(1)}psi | {log.vibration.toFixed(1)}hz
                    </td>
                    <td className="py-3.5 px-4">
                      {log.prediction === 1 ? (
                        <span className="px-2.5 py-0.5 rounded-full bg-red-500/20 text-red-400 font-bold border border-red-500/30">
                          Failure (1)
                        </span>
                      ) : (
                        <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 font-bold border border-emerald-500/30">
                          Normal (0)
                        </span>
                      )}
                    </td>
                    <td className="py-3.5 px-4 font-mono text-white">{(log.probability * 100).toFixed(1)}%</td>
                    <td className="py-3.5 px-4 font-mono text-emerald-400">{(log.confidence * 100).toFixed(1)}%</td>
                    <td className="py-3.5 px-4 font-mono text-amber-400">{log.latency_ms ?? 35}ms</td>
                    <td className="py-3.5 px-4 font-mono text-gray-400">{log.model_version || 'v2.0.0'}</td>
                    <td className="py-3.5 px-4">
                      <span className={`font-bold ${
                        log.risk_level === 'High' ? 'text-red-400' : log.risk_level === 'Warning' ? 'text-amber-400' : 'text-emerald-400'
                      }`}>
                        {log.risk_level}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Controls */}
        <div className="flex items-center justify-between p-4 border-t border-gray-800 text-xs text-gray-400">
          <div>
            Showing <span className="font-bold text-white">{logs.length}</span> of {total} total predictions
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="p-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-white"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="font-mono">Page {page + 1} of {Math.ceil(total / limit) || 1}</span>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={(page + 1) * limit >= total}
              className="p-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-white"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
