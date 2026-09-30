import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Cpu,
  Activity,
  Zap,
  BrainCircuit,
  Wrench,
  DollarSign,
  Clock,
  Gauge,
  CheckCircle2,
  AlertTriangle,
  Flame,
  LineChart,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import CustomTooltip from './CustomTooltip';

export interface MachineDetail {
  machineID: number | string;
  model: string;
  age: number;
  status: 'Healthy' | 'Warning' | 'Critical';
  healthScore: number;
  failureProbability: number;
  confidence: number;
  volt: number;
  rotate: number;
  pressure: number;
  vibration: number;
  estDowntimeHrs: number;
  estRepairCostUsd: number;
  recommendation: string;
  telemetryHistory?: Array<{ time: string; volt: number; rotate: number; pressure: number; vibration: number }>;
}

interface MachineDetailPanelProps {
  machine: MachineDetail | null;
  isOpen: boolean;
  onClose: () => void;
  onRunPredict?: (machineId: number | string) => void;
  onExplainShap?: (machineId: number | string) => void;
}

// Generate realistic telemetry sparkline data if not provided
const generateMockSparkline = (baseVolt: number, baseRotate: number, basePressure: number, baseVib: number) => {
  const times = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', 'Now'];
  return times.map((t, idx) => ({
    time: t,
    volt: +(baseVolt + (Math.sin(idx) * 4)).toFixed(1),
    rotate: +(baseRotate + (Math.cos(idx) * 12)).toFixed(0),
    pressure: +(basePressure + (Math.sin(idx * 2) * 3)).toFixed(1),
    vibration: +(baseVib + (idx * 1.5)).toFixed(1),
  }));
};

export default function MachineDetailPanel({
  machine,
  isOpen,
  onClose,
  onRunPredict,
  onExplainShap,
}: MachineDetailPanelProps) {
  if (!machine) return null;

  const sparklineData =
    machine.telemetryHistory ||
    generateMockSparkline(machine.volt, machine.rotate, machine.pressure, machine.vibration);

  const isCritical = machine.status === 'Critical' || machine.failureProbability >= 0.7;
  const isWarning = machine.status === 'Warning' || (machine.failureProbability >= 0.3 && machine.failureProbability < 0.7);

  const statusColor = isCritical ? 'text-rose-400' : isWarning ? 'text-amber-400' : 'text-emerald-400';
  const statusBg = isCritical
    ? 'bg-rose-500/10 border-rose-500/20 text-rose-400'
    : isWarning
    ? 'bg-amber-500/10 border-amber-500/20 text-amber-400'
    : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400';

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
          />

          {/* Drawer Panel */}
          <motion.aside
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="fixed right-0 top-0 z-50 h-screen w-full max-w-md bg-[#0b0c13]/95 border-l border-white/[0.08] backdrop-blur-2xl flex flex-col shadow-2xl overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-5 border-b border-white/[0.08] bg-white/[0.02]">
              <div className="flex items-center gap-3">
                <div className={`p-2.5 rounded-xl border ${statusBg}`}>
                  <Cpu className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                    Machine M-{machine.machineID}
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase tracking-wider ${statusBg}`}>
                      {machine.status}
                    </span>
                  </h2>
                  <p className="text-[11px] text-gray-500 font-medium mt-0.5">
                    Model {machine.model.toUpperCase()} • Operating Age {machine.age} Years
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] text-gray-400 hover:text-white flex items-center justify-center transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Scrollable Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
              {/* Health Score & Failure Risk Ring */}
              <div className="glass-card p-5 bg-gradient-to-br from-white/[0.03] to-transparent">
                <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider mb-4">
                  Diagnostic Overview
                </p>

                <div className="grid grid-cols-2 gap-4 items-center">
                  {/* Circular Health Gauge */}
                  <div className="flex flex-col items-center justify-center">
                    <div className="relative w-28 h-28 flex items-center justify-center">
                      <svg className="w-full h-full transform -rotate-90">
                        <circle cx="56" cy="56" r="46" stroke="rgba(255,255,255,0.05)" strokeWidth="8" fill="transparent" />
                        <motion.circle
                          cx="56"
                          cy="56"
                          r="46"
                          stroke={isCritical ? '#f43f5e' : isWarning ? '#f59e0b' : '#10b981'}
                          strokeWidth="8"
                          fill="transparent"
                          strokeDasharray={289}
                          initial={{ strokeDashoffset: 289 }}
                          animate={{ strokeDashoffset: 289 - (289 * (machine.healthScore / 100)) }}
                          transition={{ duration: 1, ease: 'easeOut' }}
                        />
                      </svg>
                      <div className="text-center z-10">
                        <span className="text-2xl font-extrabold text-white">{machine.healthScore}</span>
                        <p className="text-[9px] text-gray-500 font-bold uppercase tracking-wider">Health Index</p>
                      </div>
                    </div>
                  </div>

                  {/* Failure Prob & Confidence */}
                  <div className="space-y-3 border-l border-white/[0.06] pl-4">
                    <div>
                      <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Failure Probability</span>
                      <p className={`text-xl font-extrabold ${statusColor}`}>
                        {(machine.failureProbability * 100).toFixed(1)}%
                      </p>
                    </div>

                    <div>
                      <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Prediction Confidence</span>
                      <p className="text-sm font-bold text-white">
                        {(machine.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Sensor Telemetry Gauges */}
              <div className="space-y-3">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider flex items-center gap-2">
                  <Gauge className="w-4 h-4 text-cyan-400" /> Current Sensor Readings
                </h3>

                <div className="grid grid-cols-2 gap-3">
                  {/* Volt */}
                  <div className="glass-card p-3 space-y-1">
                    <div className="flex justify-between items-center text-[11px]">
                      <span className="text-gray-400 font-medium">Voltage</span>
                      <span className="text-white font-bold">{machine.volt.toFixed(1)} V</span>
                    </div>
                    <div className="progress-bar">
                      <div
                        className="fill bg-gradient-to-r from-indigo-500 to-cyan-400"
                        style={{ width: `${Math.min(100, (machine.volt / 250) * 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* Rotate */}
                  <div className="glass-card p-3 space-y-1">
                    <div className="flex justify-between items-center text-[11px]">
                      <span className="text-gray-400 font-medium">Rotation</span>
                      <span className="text-white font-bold">{machine.rotate.toFixed(0)} rpm</span>
                    </div>
                    <div className="progress-bar">
                      <div
                        className="fill bg-gradient-to-r from-purple-500 to-factory-400"
                        style={{ width: `${Math.min(100, (machine.rotate / 600) * 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* Pressure */}
                  <div className="glass-card p-3 space-y-1">
                    <div className="flex justify-between items-center text-[11px]">
                      <span className="text-gray-400 font-medium">Pressure</span>
                      <span className="text-white font-bold">{machine.pressure.toFixed(1)} psi</span>
                    </div>
                    <div className="progress-bar">
                      <div
                        className="fill bg-gradient-to-r from-amber-500 to-orange-400"
                        style={{ width: `${Math.min(100, (machine.pressure / 160) * 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* Vibration */}
                  <div className="glass-card p-3 space-y-1">
                    <div className="flex justify-between items-center text-[11px]">
                      <span className="text-gray-400 font-medium">Vibration</span>
                      <span className="text-white font-bold">{machine.vibration.toFixed(1)} mm/s</span>
                    </div>
                    <div className="progress-bar">
                      <div
                        className="fill bg-gradient-to-r from-rose-500 to-pink-500"
                        style={{ width: `${Math.min(100, (machine.vibration / 70) * 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Historical Telemetry Sparkline */}
              <div className="glass-card p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-1.5">
                    <LineChart className="w-3.5 h-3.5 text-factory-400" /> Vibration Trend (24h)
                  </h4>
                  <span className="text-[10px] text-gray-500 font-medium">Live Telemetry Stream</span>
                </div>

                <div className="h-32 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={sparklineData}>
                      <defs>
                        <linearGradient id="vibColor" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.4} />
                          <stop offset="95%" stopColor="#f43f5e" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="time" stroke="#475569" tick={{ fontSize: 9 }} />
                      <YAxis hide domain={['auto', 'auto']} />
                      <Tooltip content={<CustomTooltip unit="mm/s" />} />
                      <Area type="monotone" dataKey="vibration" stroke="#f43f5e" strokeWidth={2} fillOpacity={1} fill="url(#vibColor)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Maintenance & Impact Estimate */}
              <div className="grid grid-cols-2 gap-3">
                <div className="glass-card p-3 flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
                    <Clock className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-gray-500 uppercase">Est. Downtime</span>
                    <p className="text-xs font-bold text-white">{machine.estDowntimeHrs || 24} Hours</p>
                  </div>
                </div>

                <div className="glass-card p-3 flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
                    <DollarSign className="w-4 h-4" />
                  </div>
                  <div>
                    <span className="text-[10px] font-bold text-gray-500 uppercase">Est. Repair Cost</span>
                    <p className="text-xs font-bold text-white">${(machine.estRepairCostUsd || 8500).toLocaleString()}</p>
                  </div>
                </div>
              </div>

              {/* Maintenance Protocol Recommendation */}
              <div className="glass-card p-4 space-y-2 border-l-2 border-l-factory-500">
                <div className="flex items-center gap-2">
                  <Wrench className="w-4 h-4 text-factory-400" />
                  <span className="text-xs font-bold text-white">Recommended Maintenance Action</span>
                </div>
                <p className="text-xs text-gray-300 leading-relaxed font-normal">
                  {machine.recommendation ||
                    (isCritical
                      ? 'Schedule immediate bearing and seal replacement. High failure probability detected.'
                      : isWarning
                      ? 'Inspect sensor calibration and conduct routine lubrication within 48 hours.'
                      : 'Continue regular operating protocol. Telemetry parameters within normal bounds.')}
                </p>
              </div>
            </div>

            {/* Footer Action Buttons */}
            <div className="p-5 border-t border-white/[0.08] bg-[#0c0d14] grid grid-cols-2 gap-3">
              <button
                onClick={() => onRunPredict?.(machine.machineID)}
                className="h-10 rounded-xl bg-gradient-to-r from-factory-500 to-purple-600 text-white font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-factory-500/20 hover:opacity-90 active:scale-[0.98] transition-all"
              >
                <Zap className="w-3.5 h-3.5" /> Run Diagnostic
              </button>

              <button
                onClick={() => onExplainShap?.(machine.machineID)}
                className="h-10 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] text-gray-200 font-bold text-xs flex items-center justify-center gap-2 active:scale-[0.98] transition-all"
              >
                <BrainCircuit className="w-3.5 h-3.5 text-cyan-400" /> Explain SHAP
              </button>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
