/**
 * Predictions.tsx
 * ---------------
 * Real-time machine failure prediction workspace with live SHAP explainability
 * and operational maintenance decision telemetry.
 *
 * Production-readiness additions
 * --------------------------------
 * - Wrapped in <ErrorBoundary> — uncaught render errors show a recovery card.
 * - <LoadingSpinner> for machine list and telemetry fetching.
 * - In-memory machine features cache for instant responsiveness without redundant network queries.
 * - useQuery with stable fetcher identity and deps: [selectedMachineId] prevents refetch loops.
 * - Real live XGBoost 2.0.0 inference (/api/predict) and live SHAP decomposition (/api/explain).
 * - Full operational diagnostic report: Risk Score, Health Score, Risk Tier, Priority, Urgency,
 *   Recommended Action, and Primary SHAP Driver waterfall.
 * - Client-side validation guards against submitting invalid sensor ranges.
 * - Backend-unavailable banner displayed when health check fails.
 * - Retry handler re-runs the last prediction payload.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from '../hooks/useQuery';
import {
  fetchMachines,
  fetchMachineFeatures,
  postPrediction,
  postExplain,
  fetchHealth,
  ApiError as ApiErrorClass,
} from '../services/api';
import {
  Zap,
  Settings,
  RefreshCw,
  AlertTriangle,
  ShieldCheck,
  Thermometer,
  WifiOff,
  Activity,
  Layers,
  Sparkles,
} from 'lucide-react';
import ErrorBoundary from '../components/ui/ErrorBoundary';
import ApiError from '../components/ui/ApiError';
import LoadingSpinner from '../components/ui/LoadingSpinner';

// ─── Sensor validation bounds (mirrors backend validators.py) ─────────────────
const SENSOR_BOUNDS = {
  volt:      { min: 0,  max: 400,  label: 'Voltage',        unit: 'V'    },
  rotate:    { min: 0,  max: 1500, label: 'Rotation Speed', unit: 'rpm'  },
  pressure:  { min: 0,  max: 400,  label: 'Pressure',       unit: 'psi'  },
  vibration: { min: 0,  max: 200,  label: 'Vibration',      unit: 'mm/s' },
};

function validateSensors(
  volt: number,
  rotate: number,
  pressure: number,
  vibration: number,
): string | null {
  const checks: Array<[number, keyof typeof SENSOR_BOUNDS]> = [
    [volt, 'volt'],
    [rotate, 'rotate'],
    [pressure, 'pressure'],
    [vibration, 'vibration'],
  ];
  for (const [val, key] of checks) {
    const { min, max, label, unit } = SENSOR_BOUNDS[key];
    if (val < min || val > max) {
      return `${label} must be between ${min} and ${max} ${unit}. Current value: ${val.toFixed(1)}.`;
    }
  }
  return null;
}

// ─── Decision helpers (mirrors backend decision_engine_service.py) ────────────
function calculateHealthScore(failureProbability: number): number {
  const rawScore = 100.0 * (1.0 - failureProbability);
  return Math.max(0.0, Math.min(100.0, Math.round(rawScore * 10) / 10));
}

function determineRiskTier(failureProbability: number): 'CRITICAL' | 'WARNING' | 'MONITOR' {
  if (failureProbability >= 0.50) return 'CRITICAL';
  if (failureProbability >= 0.25) return 'WARNING';
  return 'MONITOR';
}

function determinePriority(riskTier: 'CRITICAL' | 'WARNING' | 'MONITOR'): string {
  switch (riskTier) {
    case 'CRITICAL': return 'P1 - Critical';
    case 'WARNING':  return 'P2 - High';
    case 'MONITOR':  return 'P3 - Low';
  }
}

function determineUrgency(riskTier: 'CRITICAL' | 'WARNING' | 'MONITOR'): string {
  switch (riskTier) {
    case 'CRITICAL': return 'Immediate';
    case 'WARNING':  return 'Scheduled';
    case 'MONITOR':  return 'Routine';
  }
}

// ─── Inner page ───────────────────────────────────────────────────────────────

function PredictionsPage() {
  const [selectedMachineId, setSelectedMachineId] = useState<number>(1);
  const [volt,      setVolt]      = useState<number>(170.0);
  const [rotate,    setRotate]    = useState<number>(450.0);
  const [pressure,  setPressure]  = useState<number>(100.0);
  const [vibration, setVibration] = useState<number>(40.0);
  const [machineAge,   setMachineAge]   = useState<number>(10);
  const [machineModel, setMachineModel] = useState<string>('model3');
  const [clientError,  setClientError]  = useState<string | null>(null);

  // Diagnostics execution state
  const [isInferring, setIsInferring] = useState<boolean>(false);
  const [predictionData, setPredictionData] = useState<any | null>(null);
  const [explainData, setExplainData] = useState<any | null>(null);
  const [diagnosticError, setDiagnosticError] = useState<any | null>(null);

  // Features cache across machines to avoid repeated fetching
  const featuresCache = useRef<Map<number, any>>(new Map());
  const lastPayload = useRef<Record<string, any> | null>(null);

  // ── Health check ──────────────────────────────────────────────────────────
  const { data: healthData, error: healthError } = useQuery(fetchHealth, {
    refetchInterval: 30000,
  });
  const backendDown =
    healthError !== null ||
    (healthData && !healthData.success && (healthData as any).status === 'degraded');

  // ── Machine list ──────────────────────────────────────────────────────────
  const { data: machinesData, isLoading: isListLoading, error: machinesError } =
    useQuery(fetchMachines);
  const machines = (machinesData as any)?.data?.machines || (machinesData as any)?.machines || [];

  // ── Feature fetch (cache-aware, stable fetcher identity) ────────────────
  const stableFetchFeatures = useCallback(
    () => {
      // Return cached data immediately if available (no network call)
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

  // Synchronize telemetry with features
  useEffect(() => {
    // featureData may arrive as { data: { features: {...} } } OR { features: {...} } OR raw
    const raw = (featureData as any)?.data?.features || (featureData as any)?.features || featureData;
    if (raw && typeof raw === 'object' && 'volt' in raw) {
      featuresCache.current.set(selectedMachineId, raw);
      setVolt(Number(raw.volt));
      setRotate(Number(raw.rotate));
      setPressure(Number(raw.pressure));
      setVibration(Number(raw.vibration));
      setMachineAge(Number(raw.age));
      setMachineModel(String(raw.model));
    }
  }, [featureData, selectedMachineId]);

  // Handle machine selection change
  const handleMachineSelect = (mId: number) => {
    setSelectedMachineId(mId);
    setPredictionData(null);
    setExplainData(null);
    setDiagnosticError(null);
    setClientError(null);

    // If cached, immediately set sensor values
    if (featuresCache.current.has(mId)) {
      const cached = featuresCache.current.get(mId);
      setVolt(Number(cached.volt));
      setRotate(Number(cached.rotate));
      setPressure(Number(cached.pressure));
      setVibration(Number(cached.vibration));
      setMachineAge(Number(cached.age));
      setMachineModel(String(cached.model));
    }
  };

  // ── Inference & Explanation execution ─────────────────────────────────────
  const executeDiagnostics = async (payload: Record<string, any>) => {
    setIsInferring(true);
    setDiagnosticError(null);

    try {
      // Execute both real prediction and real SHAP explanation concurrently
      const [predRes, expRes] = await Promise.all([
        postPrediction(payload),
        postExplain(payload).catch((expErr) => {
          console.warn('[Predictions] SHAP explanation fallback:', expErr);
          return null;
        }),
      ]);

      const pData = (predRes as any)?.data || predRes;
      const eData = (expRes as any)?.data || expRes;

      setPredictionData(pData);
      setExplainData(eData);
    } catch (err: any) {
      console.error('[Predictions] Diagnostics execution error:', err);
      setDiagnosticError(err);
    } finally {
      setIsInferring(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setClientError(null);

    const validationMsg = validateSensors(volt, rotate, pressure, vibration);
    if (validationMsg) {
      setClientError(validationMsg);
      return;
    }

    const payload = { machineID: selectedMachineId, volt, rotate, pressure, vibration };
    lastPayload.current = payload;
    executeDiagnostics(payload);
  };

  const handleRetry = () => {
    if (lastPayload.current) {
      executeDiagnostics(lastPayload.current);
    }
  };

  // Calculated decision metrics
  const failureProb = predictionData ? Number(predictionData.probability || 0) : 0;
  const healthScore = calculateHealthScore(failureProb);
  const riskTier = determineRiskTier(failureProb);
  const priority = determinePriority(riskTier);
  const urgency = determineUrgency(riskTier);

  // SHAP contributions and primary driver
  const contributions: Array<{ feature: string; shap_value: number }> =
    explainData?.contributions || [];
  const primaryDriver = contributions.length > 0
    ? [...contributions].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))[0]
    : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-extrabold text-white tracking-tight">Failure Predictions</h1>
        <p className="text-xs text-gray-500 mt-1">
          Adjust real-time sensor inputs and run predictive maintenance diagnostic simulations with live SHAP explanations.
        </p>
      </div>

      {/* Backend unavailable banner */}
      <AnimatePresence>
        {backendDown && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="flex items-center gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400"
          >
            <WifiOff className="w-4 h-4 flex-shrink-0" />
            <p className="text-xs font-semibold">
              Backend server is unreachable. Prediction requests will fail until the Flask API is restored.
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Input parameters */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="glass-card p-5 lg:col-span-2"
        >
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-sm font-bold text-white tracking-wider uppercase flex items-center gap-2">
              <Settings className="w-4 h-4 text-factory-400" />
              Sensor Tuning Workspace
            </h2>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider bg-white/[0.04] px-2.5 py-1 rounded-lg border border-white/[0.08]">
                {machineModel.toUpperCase()} • Age: {machineAge} yrs
              </span>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Machine Selector */}
            <div>
              <label className="block text-[11px] font-bold uppercase tracking-wider text-gray-500 mb-2">
                Target Machine ID
              </label>
              {isListLoading ? (
                <LoadingSpinner size="sm" label="Loading machines..." />
              ) : machinesError ? (
                <ApiError error={machinesError} />
              ) : (
                <select
                  value={selectedMachineId}
                  onChange={(e) => handleMachineSelect(Number(e.target.value))}
                  className="w-full h-10 px-4 rounded-xl bg-white/[0.04] border border-white/[0.08] text-sm text-gray-300 focus:outline-none focus:border-factory-500 focus:ring-1 focus:ring-factory-500/20 transition-all cursor-pointer"
                >
                  {machines.map((m: any) => (
                    <option key={m.machineID} value={m.machineID} className="bg-[#111219] text-gray-300">
                      Machine ID: {m.machineID} ({m.model.toUpperCase()} • Age {m.age} Yrs)
                    </option>
                  ))}
                </select>
              )}
            </div>

            {isFeaturesLoading && !featuresCache.current.has(selectedMachineId) ? (
              <LoadingSpinner size="md" label="Loading sensor data..." fullPage />
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Volt Slider */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-gray-400 font-medium">Voltage (volt)</span>
                    <span className="text-white font-bold">{volt.toFixed(1)} V</span>
                  </div>
                  <input
                    type="range"
                    min="100"
                    max="250"
                    step="0.1"
                    value={volt}
                    onChange={(e) => setVolt(Number(e.target.value))}
                    className="w-full accent-factory-500 h-1 rounded-full bg-white/10"
                  />
                  <div className="flex justify-between text-[10px] text-gray-600 font-semibold uppercase">
                    <span>100 V</span><span>250 V</span>
                  </div>
                </div>

                {/* Rotate Slider */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-gray-400 font-medium">Rotation Speed (rotate)</span>
                    <span className="text-white font-bold">{rotate.toFixed(0)} rpm</span>
                  </div>
                  <input
                    type="range"
                    min="200"
                    max="600"
                    step="1"
                    value={rotate}
                    onChange={(e) => setRotate(Number(e.target.value))}
                    className="w-full accent-factory-500 h-1 rounded-full bg-white/10"
                  />
                  <div className="flex justify-between text-[10px] text-gray-600 font-semibold uppercase">
                    <span>200 rpm</span><span>600 rpm</span>
                  </div>
                </div>

                {/* Pressure Slider */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-gray-400 font-medium">Pressure (pressure)</span>
                    <span className="text-white font-bold">{pressure.toFixed(1)} psi</span>
                  </div>
                  <input
                    type="range"
                    min="60"
                    max="160"
                    step="0.1"
                    value={pressure}
                    onChange={(e) => setPressure(Number(e.target.value))}
                    className="w-full accent-factory-500 h-1 rounded-full bg-white/10"
                  />
                  <div className="flex justify-between text-[10px] text-gray-600 font-semibold uppercase">
                    <span>60 psi</span><span>160 psi</span>
                  </div>
                </div>

                {/* Vibration Slider */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-gray-400 font-medium">Vibration (vibration)</span>
                    <span className="text-white font-bold">{vibration.toFixed(1)} mm/s</span>
                  </div>
                  <input
                    type="range"
                    min="20"
                    max="70"
                    step="0.1"
                    value={vibration}
                    onChange={(e) => setVibration(Number(e.target.value))}
                    className="w-full accent-factory-500 h-1 rounded-full bg-white/10"
                  />
                  <div className="flex justify-between text-[10px] text-gray-600 font-semibold uppercase">
                    <span>20 mm/s</span><span>70 mm/s</span>
                  </div>
                </div>
              </div>
            )}

            {/* Client-side validation error */}
            <AnimatePresence>
              {clientError && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="flex items-start gap-2 p-3 rounded-xl bg-orange-500/10 border border-orange-500/20 text-orange-400"
                >
                  <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <p className="text-xs font-medium">{clientError}</p>
                </motion.div>
              )}
            </AnimatePresence>

            <button
              id="predict-submit-btn"
              type="submit"
              disabled={isFeaturesLoading || isInferring}
              className="w-full h-11 rounded-xl bg-gradient-to-r from-factory-500 to-purple-600 text-white font-bold text-sm hover:shadow-lg hover:shadow-factory-500/20 active:scale-[0.99] transition-all disabled:opacity-50 flex items-center justify-center gap-2 cursor-pointer"
            >
              {isInferring ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running Failure Diagnostics...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  Execute Diagnostics
                </>
              )}
            </button>
          </form>
        </motion.div>

        {/* Right: Results panel */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="glass-card p-5 flex flex-col justify-between"
        >
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-bold text-white tracking-wider uppercase flex items-center gap-2">
                <Activity className="w-4 h-4 text-factory-400" />
                Diagnostic Report
              </h2>
              {predictionData && (
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-factory-500/10 text-factory-400 border border-factory-500/20">
                  ID: {predictionData.prediction_id ? predictionData.prediction_id.slice(0, 12) : 'LIVE'}
                </span>
              )}
            </div>

            <AnimatePresence mode="wait">
              {diagnosticError && !isInferring ? (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                >
                  <ApiError error={diagnosticError} onRetry={handleRetry} />
                </motion.div>
              ) : predictionData ? (
                <motion.div
                  key="results"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="space-y-4"
                >
                  {/* Probability Gauge */}
                  <div className="flex flex-col items-center py-2">
                    <div className="relative flex items-center justify-center w-32 h-32">
                      <svg className="absolute w-full h-full transform -rotate-90">
                        <circle
                          cx="64"
                          cy="64"
                          r="52"
                          stroke="rgba(255,255,255,0.03)"
                          strokeWidth="8"
                          fill="transparent"
                        />
                        <motion.circle
                          cx="64"
                          cy="64"
                          r="52"
                          stroke={predictionData.prediction === 1 ? '#f43f5e' : '#10b981'}
                          strokeWidth="8"
                          fill="transparent"
                          strokeDasharray={326}
                          initial={{ strokeDashoffset: 326 }}
                          animate={{ strokeDashoffset: 326 - (326 * failureProb) }}
                          transition={{ duration: 0.8, ease: 'easeOut' }}
                        />
                      </svg>
                      <div className="text-center z-10">
                        <span className="text-2xl font-black text-white">
                          {(failureProb * 100).toFixed(0)}%
                        </span>
                        <p className="text-[9px] text-gray-500 font-bold uppercase tracking-wider mt-0.5">
                          Risk Score
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Status Box */}
                  <div className={`p-3.5 rounded-xl border flex items-center gap-3 ${
                    predictionData.prediction === 1
                      ? 'bg-rose-500/10 border-rose-500/20 text-rose-400'
                      : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                  }`}>
                    {predictionData.prediction === 1 ? (
                      <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                    ) : (
                      <ShieldCheck className="w-5 h-5 flex-shrink-0" />
                    )}
                    <div className="flex-1">
                      <p className="text-xs font-bold uppercase tracking-wider">
                        {predictionData.prediction === 1
                          ? 'High Risk — Failure Likely'
                          : 'Stable — Operations Safe'}
                      </p>
                      <p className="text-[10px] text-gray-400 mt-0.5">
                        Confidence: {(Number(predictionData.confidence || 0) * 100).toFixed(1)}% • Latency: {predictionData.latency_ms || 0}ms
                      </p>
                    </div>
                  </div>

                  {/* Operational Decision Metrics (Health, Risk Tier, Priority) */}
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.06]">
                      <span className="text-[9px] font-bold text-gray-500 uppercase tracking-wider block mb-1">
                        Health Score
                      </span>
                      <span className={`text-sm font-black ${
                        healthScore >= 75 ? 'text-emerald-400' : healthScore >= 50 ? 'text-amber-400' : 'text-rose-400'
                      }`}>
                        {healthScore.toFixed(1)}%
                      </span>
                    </div>

                    <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.06]">
                      <span className="text-[9px] font-bold text-gray-500 uppercase tracking-wider block mb-1">
                        Risk Tier
                      </span>
                      <span className={`text-xs font-black uppercase ${
                        riskTier === 'CRITICAL' ? 'text-rose-400' : riskTier === 'WARNING' ? 'text-amber-400' : 'text-emerald-400'
                      }`}>
                        {riskTier}
                      </span>
                    </div>

                    <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.06]">
                      <span className="text-[9px] font-bold text-gray-500 uppercase tracking-wider block mb-1">
                        Priority
                      </span>
                      <span className={`text-xs font-bold ${
                        priority.startsWith('P1') ? 'text-rose-400' : priority.startsWith('P2') ? 'text-amber-400' : 'text-gray-300'
                      }`}>
                        {priority}
                      </span>
                    </div>
                  </div>

                  {/* Recommended Action */}
                  <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/[0.06] space-y-1.5">
                    <div className="flex items-center justify-between">
                      <p className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">
                        Recommended Protocol
                      </p>
                      <span className="text-[9px] font-semibold text-gray-400 uppercase">
                        Urgency: {urgency}
                      </span>
                    </div>
                    <p className="text-xs text-gray-300 font-medium leading-relaxed">
                      {predictionData.recommended_action}
                    </p>
                  </div>

                  {/* SHAP Explanation Section */}
                  {contributions.length > 0 && (
                    <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/[0.06] space-y-2.5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5">
                          <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                          <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                            SHAP Feature Explanations
                          </p>
                        </div>
                        {primaryDriver && (
                          <span className={`text-[9px] font-bold px-2 py-0.5 rounded-md ${
                            primaryDriver.shap_value > 0
                              ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                              : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          }`}>
                            Primary: {primaryDriver.feature}
                          </span>
                        )}
                      </div>

                      {/* Contribution Bars */}
                      <div className="space-y-2 max-h-[140px] overflow-y-auto pr-1">
                        {contributions.slice(0, 4).map((item, idx) => {
                          const isPositive = item.shap_value > 0;
                          const pct = Math.min(Math.abs(item.shap_value) * 150, 100);
                          return (
                            <div key={idx} className="space-y-0.5">
                              <div className="flex items-center justify-between text-[11px]">
                                <span className="text-gray-300 font-medium truncate max-w-[170px]">
                                  {item.feature}
                                </span>
                                <span className={`font-bold ${isPositive ? 'text-rose-400' : 'text-emerald-400'}`}>
                                  {isPositive ? '+' : ''}{item.shap_value.toFixed(4)}
                                </span>
                              </div>
                              <div className="flex h-1.5 rounded-full overflow-hidden bg-white/5">
                                <div className="w-1/2 flex justify-end">
                                  {!isPositive && (
                                    <div
                                      style={{ width: `${pct}%` }}
                                      className="bg-emerald-500/50 h-full rounded-l-full"
                                    />
                                  )}
                                </div>
                                <div className="w-1/2 flex justify-start">
                                  {isPositive && (
                                    <div
                                      style={{ width: `${pct}%` }}
                                      className="bg-rose-500/50 h-full rounded-r-full"
                                    />
                                  )}
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Legend */}
                      <div className="flex items-center justify-between text-[9px] text-gray-500 font-semibold pt-1 border-t border-white/[0.04]">
                        <span className="text-emerald-400/80">◀ Decreases Risk</span>
                        <span className="text-rose-400/80">Increases Risk ▶</span>
                      </div>
                    </div>
                  )}

                  {/* SHAP unavailable notice */}
                  {predictionData.explanation_available === false && (
                    <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20">
                      <p className="text-[10px] font-semibold text-amber-400">
                        ℹ Explanation unavailable. Model inference succeeded.
                      </p>
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="py-20 text-center space-y-3"
                >
                  <Thermometer className="w-10 h-10 text-gray-600 mx-auto" />
                  <p className="text-sm text-gray-400 font-medium">Ready for Diagnostic Input</p>
                  <p className="text-xs text-gray-600 max-w-xs mx-auto leading-relaxed">
                    Select a machine, adjust the telemetry sliders, and click &ldquo;Execute Diagnostics&rdquo;
                    to run ML inference with live SHAP explanations.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="text-[10px] text-gray-600 text-center font-semibold uppercase tracking-wider border-t border-white/[0.04] pt-3 mt-4">
            Inference engine • {predictionData?.model_algorithm || 'XGBoost'} v{predictionData?.model_version || '2.0.0'} • {predictionData?.feature_count || 31} Features
          </div>
        </motion.div>
      </div>
    </div>
  );
}

// ─── Export wrapped in ErrorBoundary ──────────────────────────────────────────

export default function Predictions() {
  return (
    <ErrorBoundary>
      <PredictionsPage />
    </ErrorBoundary>
  );
}
