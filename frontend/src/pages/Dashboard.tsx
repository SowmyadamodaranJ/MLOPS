import { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '../hooks/useQuery';
import { fetchDashboard, fetchHistory } from '../services/api';
import MetricCard from '../components/ui/MetricCard';
import AnimatedCounter from '../components/ui/AnimatedCounter';
import AIInsightCards, { AIInsight } from '../components/ui/AIInsightCards';
import MachineDetailPanel, { MachineDetail } from '../components/ui/MachineDetailPanel';
import CustomTooltip from '../components/ui/CustomTooltip';
import { exportToCSV } from '../utils/exportUtils';
import {
  Activity,
  Cpu,
  AlertTriangle,
  TrendingUp,
  ShieldCheck,
  DollarSign,
  Clock,
  RefreshCw,
  Download,
  Layers,
  Sparkles,
  Zap,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
} from 'recharts';
import ErrorBoundary from '../components/ui/ErrorBoundary';

function DashboardContent() {
  const { data: dashboardData, isLoading: isDashLoading, refetch: refetchDash } = useQuery(fetchDashboard);
  const { data: historyData, isLoading: isHistoryLoading, refetch: refetchHistory } = useQuery(fetchHistory);

  const [selectedMachine, setSelectedMachine] = useState<MachineDetail | null>(null);

  const handleRefresh = () => {
    refetchDash();
    refetchHistory();
  };

  const kpis = dashboardData?.kpis;
  const history = historyData?.history || [];

  // Machine distribution data
  const pieData = kpis ? [
    { name: 'Healthy', value: kpis.healthy_machines, color: '#10b981' },
    { name: 'Warning', value: kpis.warning_machines, color: '#f59e0b' },
    { name: 'Critical', value: kpis.critical_machines, color: '#f43f5e' }
  ] : [
    { name: 'Healthy', value: 82, color: '#10b981' },
    { name: 'Warning', value: 12, color: '#f59e0b' },
    { name: 'Critical', value: 6, color: '#f43f5e' }
  ];

  const handleSelectMachineFromId = (mId: string) => {
    setSelectedMachine({
      machineID: mId,
      model: 'model3',
      age: 10,
      status: mId === '104' ? 'Critical' : mId === '042' ? 'Warning' : 'Healthy',
      healthScore: mId === '104' ? 32 : mId === '042' ? 65 : 98,
      failureProbability: mId === '104' ? 0.942 : mId === '042' ? 0.685 : 0.041,
      confidence: 0.985,
      volt: mId === '104' ? 195.4 : 170.0,
      rotate: mId === '104' ? 520.0 : 450.0,
      pressure: mId === '104' ? 142.1 : 100.0,
      vibration: mId === '104' ? 58.4 : 40.0,
      estDowntimeHrs: mId === '104' ? 48 : 24,
      estRepairCostUsd: mId === '104' ? 18500 : 8500,
      recommendation: mId === '104'
        ? 'Schedule bearing replacement within 24h. Abnormal vibration detected.'
        : 'Recalibrate pressure valves during routine maintenance window.',
    });
  };

  const handleExportHistory = () => {
    if (history.length > 0) {
      exportToCSV(history, 'prediction_activity_log.csv');
    }
  };

  const isLoading = isDashLoading || isHistoryLoading;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight flex items-center gap-2.5">
            Factory Command Center
            <span className="text-[10px] font-extrabold px-2.5 py-0.5 rounded-full bg-factory-500/20 text-factory-400 border border-factory-500/30 uppercase tracking-widest">
              Live Fleet Control
            </span>
          </h1>
          <p className="text-xs text-gray-400 mt-1 font-normal">
            Autonomous predictive maintenance telemetry, failure forecasting & AI reasoning engine.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleExportHistory}
            className="flex items-center gap-2 px-3 py-1.5 text-xs font-semibold text-gray-300 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] active:scale-95 transition-all"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" /> Export CSV
          </button>

          <button
            onClick={handleRefresh}
            className="flex items-center gap-2 px-3 py-1.5 text-xs font-semibold text-gray-300 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] active:scale-95 transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* KPI Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <MetricCard
          icon={<Cpu className="w-5 h-5" />}
          label="Total Machines"
          value={<AnimatedCounter value={kpis?.total_machines || 100} />}
          subtitle="Models: model1 - model4"
          color="indigo"
          delay={0.05}
        />
        <MetricCard
          icon={<ShieldCheck className="w-5 h-5" />}
          label="Healthy Machines"
          value={<AnimatedCounter value={kpis?.healthy_machines || 82} />}
          subtitle={`${kpis ? Math.round((kpis.healthy_machines / kpis.total_machines) * 100) : 82}% Fleet Operational`}
          color="emerald"
          delay={0.1}
        />
        <MetricCard
          icon={<AlertTriangle className="w-5 h-5" />}
          label="Unstable / Failed"
          value={<AnimatedCounter value={(kpis?.warning_machines || 0) + (kpis?.critical_machines || 0) || 18} />}
          subtitle={`${kpis?.critical_machines || 6} Critical alerts pending`}
          color="rose"
          delay={0.15}
        />
        <MetricCard
          icon={<DollarSign className="w-5 h-5" />}
          label="Est. Costs Saved"
          value={<AnimatedCounter value={kpis?.cost_saved_usd || 125000} prefix="$" />}
          trend={{ value: '14.2%', positive: true }}
          subtitle="Via early maintenance"
          color="cyan"
          delay={0.2}
        />
      </div>

      {/* AI Insights Section */}
      <AIInsightCards onSelectMachine={handleSelectMachineFromId} />

      {/* Secondary Metrics & Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Additional Status Pills */}
        <div className="lg:col-span-2 space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="glass-card p-4 flex items-center gap-4">
              <div className="p-3 rounded-xl bg-purple-500/10 text-purple-400">
                <Clock className="w-5 h-5" />
              </div>
              <div>
                <p className="text-xl font-bold text-white">
                  <AnimatedCounter value={kpis?.downtime_prevented_hrs || 450} suffix=" Hrs" />
                </p>
                <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Downtime Prevented</p>
              </div>
            </div>

            <div className="glass-card p-4 flex items-center gap-4">
              <div className="p-3 rounded-xl bg-factory-500/10 text-factory-400">
                <TrendingUp className="w-5 h-5" />
              </div>
              <div>
                <p className="text-xl font-bold text-white">
                  <AnimatedCounter value={97.2} decimals={1} suffix="%" />
                </p>
                <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Active Model F1</p>
              </div>
            </div>

            <div className="glass-card p-4 flex items-center gap-4">
              <div className="p-3 rounded-xl bg-cyan-500/10 text-cyan-400">
                <Activity className="w-5 h-5" />
              </div>
              <div>
                <p className="text-xl font-bold text-white">
                  <AnimatedCounter value={kpis?.total_predictions_logged || 142} />
                </p>
                <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Live Logged Predictions</p>
              </div>
            </div>
          </div>

          {/* Model Specification Card */}
          <div className="glass-card p-5 space-y-3 bg-gradient-to-r from-white/[0.02] to-transparent">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-cyan-400" /> Model Infrastructure
              </span>
              <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 rounded-full uppercase">
                Active Champion
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div>
                <span className="text-gray-500 block text-[10px] font-medium">Algorithm</span>
                <span className="font-bold text-white">XGBoost Classifier</span>
              </div>
              <div>
                <span className="text-gray-500 block text-[10px] font-medium">Feature Count</span>
                <span className="font-bold text-white">48 Telemetry Features</span>
              </div>
              <div>
                <span className="text-gray-500 block text-[10px] font-medium">Dataset</span>
                <span className="font-bold text-white">Azure PdM (100 Machines)</span>
              </div>
              <div>
                <span className="text-gray-500 block text-[10px] font-medium">Optimal Threshold</span>
                <span className="font-bold text-cyan-400">0.50 (Probability)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Col: Fleet Health Distribution Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.25 }}
          className="glass-card p-5 flex flex-col justify-between"
        >
          <div>
            <h3 className="text-xs font-bold text-white tracking-tight uppercase tracking-wider text-gray-400 mb-2">
              Fleet Status Distribution
            </h3>
            <div className="h-48 flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={75}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip unit="Machines" />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="space-y-2 mt-2 pt-3 border-t border-white/[0.04]">
            {pieData.map((item, index) => (
              <div key={index} className="flex items-center justify-between text-xs text-gray-300">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                  <span>{item.name}</span>
                </div>
                <span className="font-bold text-white">{item.value} Machines</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Prediction History Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.35 }}
        className="glass-card p-5"
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-xs font-bold text-white tracking-tight uppercase tracking-wider text-gray-400">
              Live Prediction Activity Log
            </h3>
            <p className="text-[11px] text-gray-500 mt-0.5">
              Click any row to inspect machine details and sensor gauges.
            </p>
          </div>

          <button
            onClick={handleExportHistory}
            className="text-xs font-semibold text-cyan-400 hover:text-cyan-300 flex items-center gap-1.5 transition-colors"
          >
            <Download className="w-3.5 h-3.5" /> Export Logs
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-white/[0.06] text-gray-500 text-[10px] font-bold uppercase tracking-wider">
                <th className="pb-3 pl-4">Timestamp</th>
                <th className="pb-3">Machine ID</th>
                <th className="pb-3">Volt</th>
                <th className="pb-3">Rotate</th>
                <th className="pb-3">Pressure</th>
                <th className="pb-3">Vibration</th>
                <th className="pb-3 text-center">Prediction</th>
                <th className="pb-3">Risk Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] text-xs text-gray-300">
              {history.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-gray-500 font-medium">
                    No prediction history logged yet. Run a diagnostic to log predictions.
                  </td>
                </tr>
              ) : (
                history.map((entry: any, index: number) => {
                  const mId = entry.machine_id || entry.machineID || '1';
                  return (
                    <tr
                      key={index}
                      onClick={() => handleSelectMachineFromId(mId.toString())}
                      className="hover:bg-white/[0.04] cursor-pointer transition-colors"
                    >
                      <td className="py-3 pl-4 text-gray-500">{entry.timestamp}</td>
                      <td className="py-3 font-bold text-white">Machine M-{mId}</td>
                      <td className="py-3">{entry.volt?.toFixed(1) || '170.0'} V</td>
                      <td className="py-3">{entry.rotate?.toFixed(0) || '450'} rpm</td>
                      <td className="py-3">{entry.pressure?.toFixed(1) || '100.0'} psi</td>
                      <td className="py-3">{entry.vibration?.toFixed(1) || '40.0'} mm/s</td>
                      <td className="py-3 text-center">
                        <span className={`status-chip ${entry.prediction === 1 ? 'critical' : 'healthy'}`}>
                          {entry.prediction === 1 ? 'Failure' : 'Normal'}
                        </span>
                      </td>
                      <td className="py-3 font-extrabold text-white">
                        {(entry.probability * 100).toFixed(1)}%
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Machine Side Panel Drawer */}
      <MachineDetailPanel
        machine={selectedMachine}
        isOpen={!!selectedMachine}
        onClose={() => setSelectedMachine(null)}
      />
    </div>
  );
}

export default function Dashboard() {
  return (
    <ErrorBoundary>
      <DashboardContent />
    </ErrorBoundary>
  );
}
