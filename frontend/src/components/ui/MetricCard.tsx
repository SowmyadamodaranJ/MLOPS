import { motion } from 'framer-motion';
import type { ReactNode } from 'react';

interface MetricCardProps {
  icon: ReactNode;
  label: string;
  value: ReactNode;
  subtitle?: string;
  trend?: { value: string; positive: boolean };
  color?: 'indigo' | 'emerald' | 'amber' | 'rose' | 'cyan';
  delay?: number;
}

const colorMap = {
  indigo:  { ring: 'ring-factory-500/20', iconBg: 'bg-factory-500/10', iconText: 'text-factory-400', glow: 'shadow-factory-500/10' },
  emerald: { ring: 'ring-emerald-500/20', iconBg: 'bg-emerald-500/10', iconText: 'text-emerald-400', glow: 'shadow-emerald-500/10' },
  amber:   { ring: 'ring-amber-500/20',   iconBg: 'bg-amber-500/10',   iconText: 'text-amber-400',   glow: 'shadow-amber-500/10' },
  rose:    { ring: 'ring-rose-500/20',     iconBg: 'bg-rose-500/10',    iconText: 'text-rose-400',    glow: 'shadow-rose-500/10' },
  cyan:    { ring: 'ring-cyan-500/20',     iconBg: 'bg-cyan-500/10',    iconText: 'text-cyan-400',    glow: 'shadow-cyan-500/10' },
};

export default function MetricCard({
  icon,
  label,
  value,
  subtitle,
  trend,
  color = 'indigo',
  delay = 0,
}: MetricCardProps) {
  const c = colorMap[color];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="metric-card group"
    >
      <div className="flex items-start justify-between">
        <div className={`flex items-center justify-center w-11 h-11 rounded-xl ${c.iconBg} ${c.glow} shadow-lg`}>
          <span className={c.iconText}>{icon}</span>
        </div>
        {trend && (
          <span
            className={`text-xs font-bold px-2 py-0.5 rounded-lg ${
              trend.positive
                ? 'bg-emerald-500/10 text-emerald-400'
                : 'bg-rose-500/10 text-rose-400'
            }`}
          >
            {trend.positive ? '↑' : '↓'} {trend.value}
          </span>
        )}
      </div>
      <div className="mt-4">
        <p className="text-2xl font-extrabold text-white tracking-tight">{value}</p>
        <p className="text-xs font-medium text-gray-500 mt-1">{label}</p>
        {subtitle && (
          <p className="text-[10px] text-gray-600 mt-0.5">{subtitle}</p>
        )}
      </div>
    </motion.div>
  );
}
