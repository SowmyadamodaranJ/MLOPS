import React from 'react';

interface CustomTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: string;
  unit?: string;
}

export default function CustomTooltip({ active, payload, label, unit = '' }: CustomTooltipProps) {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card p-3 bg-[#0d0e17]/95 border border-white/10 shadow-2xl backdrop-blur-xl rounded-xl text-xs space-y-1.5 min-w-[140px]">
        {label && <p className="font-bold text-gray-300 border-b border-white/10 pb-1 mb-1">{label}</p>}
        {payload.map((entry: any, index: number) => (
          <div key={`item-${index}`} className="flex items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-1.5">
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: entry.color || entry.fill || '#818cf8' }}
              />
              <span className="text-gray-400 font-medium">{entry.name || 'Value'}:</span>
            </div>
            <span className="font-bold text-white">
              {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value} {unit}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
}
