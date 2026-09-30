/**
 * LoadingSpinner.tsx
 * ------------------
 * Consistent animated loading indicator used across all pages.
 * Replaces ad-hoc shimmer divs with a single, accessible component.
 *
 * Usage
 * -----
 * <LoadingSpinner />                         // default: medium, centered
 * <LoadingSpinner size="sm" label="Fetching machines..." />
 * <LoadingSpinner fullPage />                // absolute-centered in viewport
 */

import { motion } from 'framer-motion';

interface LoadingSpinnerProps {
  /** Visual size variant. Defaults to 'md'. */
  size?: 'sm' | 'md' | 'lg';
  /** Optional descriptive label rendered below the spinner. */
  label?: string;
  /** When true, the spinner is absolutely centered in its nearest
   *  positioned ancestor.  Useful for full-page loading states. */
  fullPage?: boolean;
}

const sizeMap = {
  sm: { ring: 'w-6 h-6',  border: 'border-2' },
  md: { ring: 'w-10 h-10', border: 'border-2' },
  lg: { ring: 'w-16 h-16', border: 'border-[3px]' },
};

export default function LoadingSpinner({
  size = 'md',
  label,
  fullPage = false,
}: LoadingSpinnerProps) {
  const { ring, border } = sizeMap[size];

  const inner = (
    <div className="flex flex-col items-center justify-center gap-3">
      {/* Dual-ring animated spinner */}
      <div className={`relative ${ring}`}>
        {/* Outer translucent ring */}
        <div
          className={`absolute inset-0 rounded-full ${border} border-factory-500/20`}
        />
        {/* Spinning gradient arc */}
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }}
          className={`absolute inset-0 rounded-full ${border} border-transparent border-t-factory-400`}
        />
      </div>

      {/* Optional label */}
      {label && (
        <p className="text-xs text-gray-500 font-medium animate-pulse">{label}</p>
      )}
    </div>
  );

  if (fullPage) {
    return (
      <div className="flex items-center justify-center min-h-[300px] w-full">
        {inner}
      </div>
    );
  }

  return inner;
}
