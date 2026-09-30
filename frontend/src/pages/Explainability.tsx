/**
 * Explainability.tsx
 * ------------------
 * SHAP feature-contribution explainability workspace.
 *
 * Production-readiness additions
 * --------------------------------
 * - <ErrorBoundary> wraps the entire page.
 * - <LoadingSpinner> during machine list / feature loading.
 * - <ApiError> with Retry when the explain call fails.
 * - Graceful "Explanation unavailable" notice when SHAP fails but the
 *   backend still returns a prediction (explanation_available: false).
 * - Never leaves the user with a blank or broken panel.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation } from '../hooks/useQuery';
import { fetchMachines, fetchMachineFeatures, postExplain } from '../services/api';
import { Cpu, BrainCircuit, RefreshCw, BarChart3, Info } from 'lucide-react';
import ErrorBoundary from '../components/ui/ErrorBoundary';
import ApiError from '../components/ui/ApiError';
import LoadingSpinner from '../components/ui/LoadingSpinner';

// ─── Inner page ───────────────────────────────────────────────────────────────

function ExplainabilityPage() {
  const [selectedMachineId, setSelectedMachineId] = useState<number>(1);
  const [volt,      setVolt]      = useState<number>(170.0);
  const [rotate,    setRotate]    = useState<number>(450.0);
  const [pressure,  setPressure]  = useState<number>(100.0);
  const [vibration, setVibration] = useState<number>(40.0);

  const lastPayload = useRef<Record<string, any> | null>(null);

  // Features cache across machines to avoid repeated fetching
  const featuresCache = useRef<Map<number, any>>(new Map());

  // ── Machine list ──────────────────────────────────────────────────────────
  const { data: machinesData, isLoading: isListLoading, error: machinesError } =
    useQuery(fetchMachines);
  const machines = (machinesData as any)?.data?.machines || (machinesData as any)?.machines || [];

  // ── Feature fetch (cache-aware, stable fetcher identity) ────────────────
  const stableFetchFeatures = useCallback(
    () => {
      const cached = featuresCache.current.get(selectedMachineId);
      if (cached) return Promise.resolve(cached);
      return fetchMachineFeatures(selectedMachineId);
    },
    [selectedMachineId],
  );

  const { data: featureData, isLoading: isFeaturesLoading } = useQuery(
    stableFetchFeatures,
    { enabled: !!selectedMachineId, deps: [selectedMachineId] },
  );

  useEffect(() => {
    const features = (featureData as any)?.data?.features || (featureData as any)?.features || featureData;
    if (features && typeof features === 'object' && 'volt' in features) {
      featuresCache.current.set(selectedMachineId, features);
      setVolt(Number(features.volt));
      setRotate(Number(features.rotate));
      setPressure(Number(features.pressure));
      setVibration(Number(features.vibration));
    }
  }, [featureData, selectedMachineId]);

  // ── Explain mutation ──────────────────────────────────────────────────────
  const {
    mutate: runExplain,
    data: explainResponse,
    isLoading: isExplaining,
    error: explainError,
    reset: resetExplain,
  } = useMutation((params: any) => postExplain(params));

  const explainData = (explainResponse as any)?.data || explainResponse;
  const contributions = explainData?.contributions || [];
  const isFallback = explainData?.status === 'fallback';
  const explanationUnavailable =
    explainData?.explanation_available === false ||
    (explainData?.status && explainData.status !== 'success');

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const payload = { machineID: selectedMachineId, volt, rotate, pressure, vibration };
    lastPayload.current = payload;
    resetExplain();
    runExplain(payload).catch(() => {});
  };

  const handleRetry = () => {
    if (lastPayload.current) {
      resetExplain();
      runExplain(lastPayload.current).catch(() => {});
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-extrabold text-white tracking-tight">AI Explainability</h1>
        <p className="text-xs text-gray-500 mt-1">
          Decompose machine learning decisions into individual feature contributions using SHAP
          (Shapley Additive exPlanations).
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Input parameters */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="glass-card p-5 lg:col-span-1"
        >
          <h2 className="text-sm font-bold text-white tracking-wider uppercase mb-5 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-factory-400" />
            Explain Workspace
          </h2>

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Machine selector */}
            <div>
              <label className="block text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-2">
                Target Machine ID
              </label>
              {isListLoading ? (
                <LoadingSpinner size="sm" label="Loading machines..." />
              ) : machinesError ? (
                <ApiError error={machinesError} />
              ) : (
                <select
                  value={selectedMachineId}
                  onChange={(e) => {
                    setSelectedMachineId(Number(e.target.value));
                    resetExplain();
                  }}
                  className="w-full h-10 px-4 rounded-xl bg-white/[0.04] border border-white/[0.08] text-sm text-gray-300 focus:outline-none focus:border-factory-500 focus:ring-1 focus:ring-factory-500/20 transition-all cursor-pointer"
                >
                  {machines.map((m: any) => (
                    <option key={m.machineID} value={m.machineID} className="bg-[#111219] text-gray-300">
                      Machine ID: {m.machineID}
                    </option>
                  ))}
                </select>
              )}
            </div>

            {isFeaturesLoading ? (
              <LoadingSpinner size="md" label="Loading sensor data..." fullPage />
            ) : (
              <div className="space-y-4">
                {/* Voltage */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Voltage (volt)</span>
                    <span className="text-white font-bold">{volt.toFixed(1)} V</span>
                  </div>
                  <input type="range" min="100" max="250" step="0.1" value={volt}
                    onChange={(e) => setVolt(Number(e.target.value))}
                    className="w-full accent-factory-500 h-1 rounded-full bg-white/10" />
                </div>

                {/* Rotation */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Rotation (rotate)</span>
                    <span className="text-white font-bold">{rotate.toFixed(0)} rpm</span>
                  </div>
                  <input type="range" min="200" max="600" step="1" value={rotate}
                    onChange={(e) => setRotate(Number(e.target.value))}
                    className="w-full accent-factory-500 h-1 rounded-full bg-white/10" />
                </div>

                {/* Pressure */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Pressure (pressure)</span>
                    <span className="text-white font-bold">{pressure.toFixed(1)} psi</span>
                  </div>
                  <input type="range" min="60" max="160" step="0.1" value={pressure}
                    onChange={(e) => setPressure(Number(e.target.value))}
                    className="w-full accent-factory-500 h-1 rounded-full bg-white/10" />
                </div>

                {/* Vibration */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-400">Vibration (vibration)</span>
                    <span className="text-white font-bold">{vibration.toFixed(1)} mm/s</span>
                  </div>
                  <input type="range" min="20" max="70" step="0.1" value={vibration}
                    onChange={(e) => setVibration(Number(e.target.value))}
                    className="w-full accent-factory-500 h-1 rounded-full bg-white/10" />
                </div>
              </div>
            )}

            <button
              id="explain-submit-btn"
              type="submit"
              disabled={isFeaturesLoading || isExplaining}
              className="w-full h-10 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] text-white font-bold text-xs hover:shadow-lg active:scale-[0.99] transition-all disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {isExplaining ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  Generating SHAP Decompositions...
                </>
              ) : (
                <>
                  <BrainCircuit className="w-3.5 h-3.5" />
                  Decompose Model Decision
                </>
              )}
            </button>
          </form>
        </motion.div>

        {/* Right: SHAP chart */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="glass-card p-5 lg:col-span-2"
        >
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-sm font-bold text-white tracking-wider uppercase">
              Local Feature Contributions (SHAP Values)
            </h2>
            {isFallback && (
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
                Rule-Based Fallback
              </span>
            )}
          </div>

          <AnimatePresence mode="wait">
            {/* Error state */}
            {explainError && !isExplaining ? (
              <motion.div
                key="error"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="py-8"
              >
                <ApiError error={explainError} onRetry={handleRetry} />
              </motion.div>
            ) : explainData ? (
              <motion.div
                key="chart"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-5"
              >
                {/* Prediction summary */}
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.06] flex items-center justify-between">
                  <div>
                    <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">
                      Failure Probability
                    </span>
                    <p className="text-2xl font-black text-white">
                      {(explainData.probability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">
                      Decision Outcome
                    </span>
                    <p className={`text-sm font-bold uppercase tracking-wider ${
                      explainData.prediction === 1 ? 'text-rose-400' : 'text-emerald-400'
                    }`}>
                      {explainData.prediction === 1 ? 'Risk Detected' : 'Stable'}
                    </p>
                  </div>
                </div>

                {/* Explanation unavailable notice */}
                {explanationUnavailable && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex items-start gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20"
                  >
                    <Info className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-xs font-bold text-amber-400">Explanation Unavailable</p>
                      <p className="text-[11px] text-gray-400 mt-0.5 leading-relaxed">
                        {explainData.explanation_message ||
                          explainData.message ||
                          'SHAP computation could not complete. Prediction is still accurate.'}
                      </p>
                    </div>
                  </motion.div>
                )}

                {/* Contribution bars */}
                {contributions.length > 0 && (
                  <div className="space-y-3 max-h-[360px] overflow-y-auto pr-2">
                    {contributions.map((item: any, index: number) => {
                      const isPositive = item.shap_value > 0;
                      const pct = Math.min(Math.abs(item.shap_value) * 150, 100);
                      return (
                        <div key={index} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-gray-300 font-medium">{item.feature}</span>
                            <span className={`font-bold ${isPositive ? 'text-rose-400' : 'text-emerald-400'}`}>
                              {isPositive ? '+' : ''}{item.shap_value.toFixed(4)}
                            </span>
                          </div>
                          <div className="progress-bar flex">
                            {isPositive ? (
                              <div className="w-1/2 flex justify-start bg-transparent">
                                <div className="w-full bg-white/5 h-full" />
                              </div>
                            ) : (
                              <div className="w-1/2 flex justify-end bg-transparent">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${pct}%` }}
                                  className="bg-emerald-500/30 h-full rounded-l-full border-r border-emerald-500/50"
                                />
                              </div>
                            )}
                            {isPositive ? (
                              <div className="w-1/2 flex justify-start bg-transparent">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${pct}%` }}
                                  className="bg-rose-500/30 h-full rounded-r-full border-l border-rose-500/50"
                                />
                              </div>
                            ) : (
                              <div className="w-1/2 flex justify-start bg-transparent">
                                <div className="w-full bg-white/5 h-full" />
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Legend */}
                <div className="flex items-center justify-center gap-6 text-[10px] text-gray-500 font-bold uppercase tracking-wider pt-2 border-t border-white/[0.04]">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/30 border border-emerald-500/50" />
                    <span>Decreases Failure Risk</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-500/30 border border-rose-500/50" />
                    <span>Increases Failure Risk</span>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="py-32 text-center space-y-4"
              >
                <BarChart3 className="w-10 h-10 text-gray-600 mx-auto" />
                <p className="text-sm text-gray-400 font-medium">
                  Ready for Explainability Decompositions
                </p>
                <p className="text-xs text-gray-600 max-w-sm mx-auto leading-relaxed">
                  Select a machine and adjust telemetry values. Click "Decompose Model Decision"
                  to generate a local SHAP contribution plot.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
}

// ─── Export wrapped in ErrorBoundary ──────────────────────────────────────────

export default function Explainability() {
  return (
    <ErrorBoundary>
      <ExplainabilityPage />
    </ErrorBoundary>
  );
}
