import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bell,
  AlertTriangle,
  AlertOctagon,
  Info,
  CheckCircle,
  RefreshCw,
  Filter,
  CheckCheck,
} from 'lucide-react';
import { fetchAlerts, postAcknowledgeAlert } from '../services/api';
import type { AlertItem } from '../types';
import LoadingSpinner from '../components/ui/LoadingSpinner';

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [severityFilter, setSeverityFilter] = useState<string>('ALL');

  const loadAlerts = async () => {
    try {
      setRefreshing(true);
      const res = await fetchAlerts();
      if (res.success && res.data) {
        setAlerts(res.data.alerts || []);
      }
    } catch (e) {
      console.error("Failed to load alerts:", e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadAlerts();
    const interval = setInterval(loadAlerts, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleAcknowledge = async (id: string) => {
    try {
      await postAcknowledgeAlert(id);
      setAlerts((prev) =>
        prev.map((a) => (a.id === id ? { ...a, acknowledged: true } : a))
      );
    } catch (e) {
      console.error("Failed to acknowledge alert:", e);
    }
  };

  const filteredAlerts = alerts.filter((a) => {
    if (severityFilter === 'ALL') return true;
    if (severityFilter === 'ACTIVE') return !a.acknowledged;
    return a.severity === severityFilter;
  });

  if (loading && alerts.length === 0) {
    return <LoadingSpinner label="Evaluating Active Automated Alerts Rules..." />;
  }

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case 'CRITICAL':
        return (
          <span className="flex items-center gap-1 text-xs font-bold text-red-400 bg-red-500/20 px-2.5 py-1 rounded-full border border-red-500/30">
            <AlertOctagon className="w-3.5 h-3.5" /> CRITICAL
          </span>
        );
      case 'HIGH':
        return (
          <span className="flex items-center gap-1 text-xs font-bold text-amber-400 bg-amber-500/20 px-2.5 py-1 rounded-full border border-amber-500/30">
            <AlertTriangle className="w-3.5 h-3.5" /> HIGH
          </span>
        );
      case 'MEDIUM':
        return (
          <span className="flex items-center gap-1 text-xs font-bold text-purple-400 bg-purple-500/20 px-2.5 py-1 rounded-full border border-purple-500/30">
            <Info className="w-3.5 h-3.5" /> MEDIUM
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1 text-xs font-bold text-cyan-400 bg-cyan-500/20 px-2.5 py-1 rounded-full border border-cyan-500/30">
            <Info className="w-3.5 h-3.5" /> INFO
          </span>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-card p-6 border-l-4 border-l-red-500">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-white tracking-tight">Enterprise Automated Alerts Feed</h1>
            <span className="px-3 py-1 text-xs font-semibold rounded-full bg-red-500/20 text-red-400 border border-red-500/30">
              MODULE 6
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Real-time rule engine for latency, drift, memory spikes, model unavailability, and DB disconnection.
          </p>
        </div>

        <button
          onClick={loadAlerts}
          disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-factory-600 hover:bg-factory-500 text-white rounded-lg transition-all"
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh Feed
        </button>
      </div>

      {/* Filters Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 glass-card p-4">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-gray-400" />
          <span className="text-xs font-bold text-gray-400 uppercase">Filter Severity:</span>
          {['ALL', 'ACTIVE', 'CRITICAL', 'HIGH', 'MEDIUM'].map((sev) => (
            <button
              key={sev}
              onClick={() => setSeverityFilter(sev)}
              className={`px-3 py-1 text-xs font-semibold rounded-lg transition-all ${
                severityFilter === sev
                  ? 'bg-factory-500 text-white shadow-md'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {sev}
            </button>
          ))}
        </div>

        <div className="text-xs text-gray-400">
          Showing <span className="font-bold text-white">{filteredAlerts.length}</span> of {alerts.length} total alerts
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-3">
        {filteredAlerts.length === 0 ? (
          <div className="glass-card p-12 text-center text-gray-400">
            <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
            <h3 className="text-lg font-bold text-white">No Active Alerts Triggered</h3>
            <p className="text-xs text-gray-500 mt-1">All monitored systems and models are operating within normal tolerances.</p>
          </div>
        ) : (
          <AnimatePresence>
            {filteredAlerts.map((alert) => (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className={`glass-card p-5 border-l-4 transition-all ${
                  alert.acknowledged
                    ? 'opacity-60 border-l-gray-600 bg-gray-950/40'
                    : alert.severity === 'CRITICAL'
                    ? 'border-l-red-500 bg-red-500/5'
                    : alert.severity === 'HIGH'
                    ? 'border-l-amber-500 bg-amber-500/5'
                    : 'border-l-purple-500 bg-purple-500/5'
                }`}
              >
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-3">
                      {getSeverityBadge(alert.severity)}
                      <span className="text-xs font-mono text-gray-400 uppercase tracking-wide">[{alert.category}]</span>
                      <h4 className="text-base font-bold text-white">{alert.title}</h4>
                    </div>
                    <p className="text-sm text-gray-300 mt-1">{alert.message}</p>
                    <p className="text-xs text-cyan-400 mt-2 font-medium">
                      Recommended Action: <span className="text-gray-300 font-normal">{alert.action}</span>
                    </p>
                  </div>

                  <div className="flex flex-col items-end gap-2 shrink-0">
                    <span className="text-xs text-gray-400 font-mono">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </span>

                    {alert.acknowledged ? (
                      <span className="flex items-center gap-1 text-xs font-semibold text-gray-400 bg-gray-800 px-3 py-1 rounded-lg">
                        <CheckCheck className="w-4 h-4 text-emerald-400" /> Acknowledged
                      </span>
                    ) : (
                      <button
                        onClick={() => handleAcknowledge(alert.id)}
                        className="px-3 py-1.5 text-xs font-semibold bg-gray-800 hover:bg-emerald-600 text-white rounded-lg transition-all border border-gray-700"
                      >
                        Acknowledge
                      </button>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}
