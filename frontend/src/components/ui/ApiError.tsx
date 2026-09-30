/**
 * ApiError.tsx
 * ------------
 * Reusable error display component for API call failures.
 *
 * Distinguishes three failure categories and renders appropriate
 * human-readable messages:
 *   1. Network / backend unavailable  (no response received)
 *   2. Validation error               (HTTP 400 VALIDATION_ERROR)
 *   3. Server error                   (HTTP 500, 503, 504, etc.)
 *
 * Usage
 * -----
 * <ApiError error={inferenceError} onRetry={() => runInference(params)} />
 */

import { motion } from 'framer-motion';
import { AlertTriangle, WifiOff, ShieldAlert, RefreshCw, ServerCrash } from 'lucide-react';

export interface ApiErrorProps {
  /** The caught Error object (or AxiosError enriched by our interceptor). */
  error: any;
  /** Optional retry handler — if provided, a Retry button is rendered. */
  onRetry?: () => void;
  /** Optional CSS class overrides on the outer wrapper. */
  className?: string;
}

// ─── Derived error shape ──────────────────────────────────────────────────────

interface ParsedError {
  category: 'network' | 'validation' | 'server' | 'unknown';
  title: string;
  message: string;
  errorCode?: string;
}

function parseError(error: any): ParsedError {
  if (!error) {
    return { category: 'unknown', title: 'Unknown Error', message: 'An unexpected error occurred.' };
  }

  // Network / timeout — axios sets error.code = 'ECONNABORTED' or no response
  if (!error.response || error.code === 'ECONNABORTED' || error.message?.includes('Network Error')) {
    return {
      category: 'network',
      title: 'Backend Unavailable',
      message:
        'Cannot reach the prediction server. Please ensure the Flask backend is running on port 5000.',
    };
  }

  // Extract standardised backend body
  const body     = error.response?.data || {};
  const errorCode: string = body.error_code || body.errorCode || '';
  const backendMsg: string = body.message || error.message || 'An error occurred.';
  const status: number = error.response?.status || 500;

  // Validation errors (400)
  if (status === 400 || errorCode === 'VALIDATION_ERROR' || errorCode.startsWith('INVALID_') || errorCode === 'MISSING_FIELD' || errorCode === 'MISSING_BODY') {
    return {
      category: 'validation',
      title: 'Invalid Input',
      message: backendMsg,
      errorCode,
    };
  }

  // Model not loaded (503)
  if (status === 503 || errorCode === 'MODEL_NOT_LOADED') {
    return {
      category: 'server',
      title: 'Model Not Ready',
      message: 'The ML model is not loaded. Check the System Status page for artifact details.',
      errorCode,
    };
  }

  // Machine not found (404)
  if (status === 404 && errorCode === 'MACHINE_NOT_FOUND') {
    return {
      category: 'validation',
      title: 'Machine Not Found',
      message: backendMsg,
      errorCode,
    };
  }

  // Timeout (504)
  if (status === 504 || errorCode === 'PREDICTION_TIMEOUT') {
    return {
      category: 'server',
      title: 'Prediction Timeout',
      message: 'The prediction took too long to compute. Please try again.',
      errorCode,
    };
  }

  // Generic server error
  return {
    category: 'server',
    title: 'Server Error',
    message: backendMsg || 'An unexpected server error occurred.',
    errorCode,
  };
}

// ─── Visual config by category ────────────────────────────────────────────────

const categoryConfig = {
  network: {
    Icon: WifiOff,
    iconClass: 'text-amber-400',
    bgClass: 'bg-amber-500/10',
    borderClass: 'border-amber-500/20',
    badgeClass: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  },
  validation: {
    Icon: ShieldAlert,
    iconClass: 'text-orange-400',
    bgClass: 'bg-orange-500/10',
    borderClass: 'border-orange-500/20',
    badgeClass: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  },
  server: {
    Icon: ServerCrash,
    iconClass: 'text-rose-400',
    bgClass: 'bg-rose-500/10',
    borderClass: 'border-rose-500/20',
    badgeClass: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
  },
  unknown: {
    Icon: AlertTriangle,
    iconClass: 'text-rose-400',
    bgClass: 'bg-rose-500/10',
    borderClass: 'border-rose-500/20',
    badgeClass: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
  },
};

// ─── Component ────────────────────────────────────────────────────────────────

export default function ApiError({ error, onRetry, className = '' }: ApiErrorProps) {
  const parsed = parseError(error);
  const cfg    = categoryConfig[parsed.category];
  const { Icon } = cfg;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      className={`rounded-2xl border p-5 ${cfg.bgClass} ${cfg.borderClass} ${className}`}
    >
      <div className="flex items-start gap-4">
        {/* Icon */}
        <div className={`p-2.5 rounded-xl ${cfg.bgClass} flex-shrink-0`}>
          <Icon className={`w-5 h-5 ${cfg.iconClass}`} />
        </div>

        {/* Text */}
        <div className="flex-1 space-y-1.5 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <p className="text-sm font-bold text-white">{parsed.title}</p>
            {parsed.errorCode && (
              <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${cfg.badgeClass}`}>
                {parsed.errorCode}
              </span>
            )}
          </div>
          <p className="text-xs text-gray-400 leading-relaxed">{parsed.message}</p>
        </div>
      </div>

      {/* Retry */}
      {onRetry && (
        <div className="mt-4 flex justify-end">
          <button
            id="api-error-retry-btn"
            onClick={onRetry}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold text-gray-300 bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] active:scale-95 transition-all"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Retry
          </button>
        </div>
      )}
    </motion.div>
  );
}
