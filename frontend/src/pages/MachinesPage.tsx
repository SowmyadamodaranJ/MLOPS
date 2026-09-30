import { useState } from 'react';
import { motion } from 'framer-motion';
import { useQuery } from '../hooks/useQuery';
import { fetchMachines } from '../services/api';
import { Factory, Search, Filter, RefreshCw, Cpu, ShieldCheck, AlertTriangle, ShieldAlert } from 'lucide-react';
import MachineDetailPanel, { MachineDetail } from '../components/ui/MachineDetailPanel';
import ErrorBoundary from '../components/ui/ErrorBoundary';
import LoadingSpinner from '../components/ui/LoadingSpinner';

function MachinesPageContent() {
  const { data, isLoading, refetch } = useQuery(fetchMachines);
  const rawMachines = (data as any)?.data?.machines || (data as any)?.machines || [];

  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [selectedMachine, setSelectedMachine] = useState<MachineDetail | null>(null);

  // Generate enriched machine list for presentation
  const machines: MachineDetail[] = rawMachines.map((m: any) => {
    const isCritical = m.status === 'Critical';
    const isWarning = m.status === 'Warning';
    const healthScore = isCritical ? 34 : isWarning ? 68 : 96;
    const failureProbability = isCritical ? 0.88 : isWarning ? 0.42 : 0.05;

    return {
      machineID: m.machineID,
      model: m.model || 'model3',
      age: m.age || 10,
      status: m.status || 'Healthy',
      healthScore,
      failureProbability,
      confidence: 0.965,
      volt: m.volt || 170.0,
      rotate: m.rotate || 450.0,
      pressure: m.pressure || 100.0,
      vibration: m.vibration || 40.0,
      estDowntimeHrs: isCritical ? 36 : isWarning ? 18 : 0,
      estRepairCostUsd: isCritical ? 14500 : isWarning ? 6200 : 0,
      recommendation: isCritical
        ? 'Schedule emergency bearing replacement. Critical vibration exceeds tolerance threshold.'
        : isWarning
        ? 'Inspect pressure seals and recalibrate valve setting within 48h window.'
        : 'Telemetry operating within optimal target range.',
    };
  });

  const filteredMachines = machines.filter((m) => {
    const matchesSearch =
      m.machineID.toString().includes(search) ||
      m.model.toLowerCase().includes(search.toLowerCase());
    const matchesStatus = statusFilter === 'all' || m.status.toLowerCase() === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-white tracking-tight">Machine Fleet Command</h1>
          <p className="text-xs text-gray-500 mt-1">
            Real-time status overview and parameter telemetry across all plant assets.
          </p>
        </div>

        <button
          onClick={() => refetch()}
          className="flex items-center gap-2 px-3 py-1.5 text-xs font-semibold text-gray-300 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] transition-all"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh Fleet
        </button>
      </div>

      {/* Filter & Search Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 glass-card p-4">
        <div className="relative w-full sm:w-72">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by Machine ID or Model..."
            className="w-full h-9 pl-9 pr-4 rounded-xl bg-white/[0.04] border border-white/[0.08] text-xs text-gray-200 placeholder-gray-500 focus:outline-none focus:border-factory-500/50 transition-all"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto overflow-x-auto text-xs">
          {['all', 'healthy', 'warning', 'critical'].map((st) => (
            <button
              key={st}
              onClick={() => setStatusFilter(st)}
              className={`px-3 py-1.5 rounded-xl font-bold uppercase tracking-wider transition-all ${
                statusFilter === st
                  ? 'bg-factory-500/20 text-factory-400 border border-factory-500/30'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-white/[0.04]'
              }`}
            >
              {st}
            </button>
          ))}
        </div>
      </div>

      {/* Fleet Cards Grid */}
      {isLoading ? (
        <LoadingSpinner size="lg" label="Loading fleet telemetry..." fullPage />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-5">
          {filteredMachines.map((m, idx) => {
            const isCritical = m.status === 'Critical';
            const isWarning = m.status === 'Warning';
            const badgeBg = isCritical
              ? 'bg-rose-500/20 text-rose-400 border-rose-500/30'
              : isWarning
              ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
              : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';

            return (
              <motion.div
                key={m.machineID}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: idx * 0.04 }}
                onClick={() => setSelectedMachine(m)}
                className="glass-card p-5 cursor-pointer hover:border-factory-500/40 hover:-translate-y-1 transition-all duration-300 space-y-4 group"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-white/[0.04] border border-white/[0.08] text-factory-400 group-hover:bg-factory-500/10 transition-colors">
                      <Cpu className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                        Machine M-{m.machineID}
                      </h3>
                      <p className="text-[10px] text-gray-500 font-medium">
                        Model {m.model.toUpperCase()} • Age {m.age}Y
                      </p>
                    </div>
                  </div>

                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase tracking-wider ${badgeBg}`}>
                    {m.status}
                  </span>
                </div>

                {/* Sensor Mini Grid */}
                <div className="grid grid-cols-2 gap-2 text-[11px] pt-2 border-t border-white/[0.04]">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Volt:</span>
                    <span className="font-bold text-gray-200">{m.volt.toFixed(1)} V</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Rotate:</span>
                    <span className="font-bold text-gray-200">{m.rotate.toFixed(0)} rpm</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Pressure:</span>
                    <span className="font-bold text-gray-200">{m.pressure.toFixed(1)} psi</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Vibration:</span>
                    <span className="font-bold text-gray-200">{m.vibration.toFixed(1)} mm/s</span>
                  </div>
                </div>

                {/* Health Meter */}
                <div className="space-y-1">
                  <div className="flex justify-between items-center text-[10px] text-gray-400">
                    <span>Health Index</span>
                    <span className="font-bold text-white">{m.healthScore}%</span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className={`fill ${
                        isCritical
                          ? 'bg-rose-500'
                          : isWarning
                          ? 'bg-amber-500'
                          : 'bg-emerald-400'
                      }`}
                      style={{ width: `${m.healthScore}%` }}
                    />
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      )}

      {/* Machine Side Detail Drawer */}
      <MachineDetailPanel
        machine={selectedMachine}
        isOpen={!!selectedMachine}
        onClose={() => setSelectedMachine(null)}
      />
    </div>
  );
}

export default function MachinesPage() {
  return (
    <ErrorBoundary>
      <MachinesPageContent />
    </ErrorBoundary>
  );
}
