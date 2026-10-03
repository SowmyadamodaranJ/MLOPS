import { motion } from 'framer-motion';
import { Brain, AlertTriangle, ArrowRight, ShieldAlert, Sparkles, Clock, Wrench } from 'lucide-react';

export interface AIInsight {
  id: string;
  machineId: string;
  severity: 'critical' | 'warning' | 'info';
  title: string;
  message: string;
  recommendedAction: string;
  timeframe: string;
  failureProb: number;
  onActionClick?: (machineId: string) => void;
}

const DEFAULT_INSIGHTS: AIInsight[] = [
  {
    id: 'insight-1',
    machineId: '94',
    severity: 'critical',
    title: 'High Error Rate & Failure Risk Detected',
    message: 'Machine M-094 exhibits critical telemetry spikes (total errors rolling 24h). Predicted failure probability is 99.9%.',
    recommendedAction: 'Schedule immediate maintenance inspection and component replacement.',
    timeframe: 'Immediate (P1)',
    failureProb: 0.999,
  },
  {
    id: 'insight-2',
    machineId: '35',
    severity: 'warning',
    title: 'Voltage Instability Trend',
    message: 'Machine M-035 voltage rolling mean exceeds baseline operating band. Failure probability increased to 45.1%.',
    recommendedAction: 'Inspect electrical subsystem & verify voltage regulator stability.',
    timeframe: 'Within 24-48 hours (P2)',
    failureProb: 0.451,
  },
  {
    id: 'insight-3',
    machineId: '1',
    severity: 'info',
    title: 'Optimal Operation — Baseline Stable',
    message: 'Machine M-001 telemetry operating smoothly within standard operational parameters. Failure risk is below 1%.',
    recommendedAction: 'Continue standard routine monitoring protocol.',
    timeframe: 'Routine check in 14 days (P4)',
    failureProb: 0.005,
  },
];

interface AIInsightCardsProps {
  insights?: AIInsight[];
  onSelectMachine?: (machineId: string) => void;
}

export default function AIInsightCards({
  insights = DEFAULT_INSIGHTS,
  onSelectMachine,
}: AIInsightCardsProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-gradient-to-r from-factory-500 to-purple-600 text-white shadow-lg shadow-factory-500/20">
            <Sparkles className="w-4 h-4 animate-pulse" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-tight uppercase tracking-wider">
              AI Command Center Insights
            </h3>
            <p className="text-[11px] text-gray-500">
              Natural language diagnostic intelligence synthesized from multi-sensor telemetry & ML model predictions.
            </p>
          </div>
        </div>
        <span className="text-[10px] font-bold text-cyan-400 bg-cyan-500/10 border border-cyan-500/20 px-2.5 py-1 rounded-full uppercase tracking-wider flex items-center gap-1.5">
          <Brain className="w-3 h-3" /> Autonomous Reasoning Active
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {insights.map((insight, idx) => {
          const isCritical = insight.severity === 'critical';
          const isWarning = insight.severity === 'warning';

          const borderColor = isCritical
            ? 'border-rose-500/30 hover:border-rose-500/50 bg-gradient-to-br from-rose-500/10 via-transparent to-transparent'
            : isWarning
            ? 'border-amber-500/30 hover:border-amber-500/50 bg-gradient-to-br from-amber-500/10 via-transparent to-transparent'
            : 'border-emerald-500/30 hover:border-emerald-500/50 bg-gradient-to-br from-emerald-500/10 via-transparent to-transparent';

          const badgeBg = isCritical
            ? 'bg-rose-500/20 text-rose-400 border-rose-500/30'
            : isWarning
            ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
            : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';

          return (
            <motion.div
              key={insight.id}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: idx * 0.1 }}
              className={`glass-card p-5 border ${borderColor} transition-all duration-300 flex flex-col justify-between group`}
            >
              <div className="space-y-3">
                {/* Header */}
                <div className="flex items-start justify-between gap-2">
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase tracking-wider ${badgeBg}`}>
                    Machine M-{insight.machineId}
                  </span>
                  <div className="flex items-center gap-1 text-[11px] text-gray-500 font-semibold">
                    <Clock className="w-3 h-3" />
                    <span>{insight.timeframe}</span>
                  </div>
                </div>

                {/* Title & Message */}
                <div>
                  <h4 className="text-xs font-bold text-white tracking-tight group-hover:text-cyan-300 transition-colors flex items-center gap-1.5">
                    {isCritical ? (
                      <ShieldAlert className="w-3.5 h-3.5 text-rose-400 flex-shrink-0" />
                    ) : isWarning ? (
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
                    ) : (
                      <Sparkles className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                    )}
                    {insight.title}
                  </h4>
                  <p className="text-xs text-gray-300 leading-relaxed mt-2 font-normal">
                    "{insight.message}"
                  </p>
                </div>
              </div>

              {/* Action protocol & trigger button */}
              <div className="pt-4 mt-4 border-t border-white/[0.06] space-y-3">
                <div className="flex items-center gap-2 text-[11px] text-gray-400">
                  <Wrench className="w-3.5 h-3.5 text-factory-400 flex-shrink-0" />
                  <span className="truncate font-medium">{insight.recommendedAction}</span>
                </div>

                <button
                  onClick={() => onSelectMachine?.(insight.machineId)}
                  className="w-full h-8 rounded-lg bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] text-xs font-semibold text-gray-200 flex items-center justify-center gap-1.5 transition-all group-hover:border-factory-500/30"
                >
                  <span>Inspect Machine M-{insight.machineId}</span>
                  <ArrowRight className="w-3.5 h-3.5 text-factory-400 group-hover:translate-x-0.5 transition-transform" />
                </button>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
