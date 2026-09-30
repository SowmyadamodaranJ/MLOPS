/**
 * ErrorBoundary.tsx
 * -----------------
 * React class component that catches uncaught JavaScript errors in any
 * child component tree and renders a styled fallback UI instead of a
 * blank screen.
 *
 * Why a class component?
 * React hooks cannot implement componentDidCatch / getDerivedStateFromError.
 * The class-based Error Boundary is the only supported pattern for this.
 *
 * Usage
 * -----
 * <ErrorBoundary>
 *   <SomePageThatMightCrash />
 * </ErrorBoundary>
 */

import React, { Component, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
  /** Optional custom fallback.  If omitted, the built-in card is rendered. */
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  errorMessage: string;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, errorMessage: '' };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render shows the fallback UI.
    return { hasError: true, errorMessage: error.message };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // Log to console for debugging; in production this would send to
    // a monitoring service (e.g. Sentry).
    console.error('[ErrorBoundary] Caught error:', error, info.componentStack);
  }

  handleReset = (): void => {
    this.setState({ hasError: false, errorMessage: '' });
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="flex items-center justify-center min-h-[400px] p-8">
          <div className="glass-card p-8 max-w-md w-full text-center space-y-5">
            {/* Icon */}
            <div className="flex justify-center">
              <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20">
                <AlertTriangle className="w-10 h-10 text-rose-400" />
              </div>
            </div>

            {/* Title */}
            <div className="space-y-2">
              <h2 className="text-lg font-bold text-white">Something went wrong</h2>
              <p className="text-sm text-gray-400 leading-relaxed">
                An unexpected error occurred in this view. Your other pages are
                unaffected.
              </p>
            </div>

            {/* Error detail — collapsed to avoid overwhelming the user */}
            {this.state.errorMessage && (
              <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.06] text-left">
                <p className="text-[10px] font-mono text-gray-500 break-all">
                  {this.state.errorMessage}
                </p>
              </div>
            )}

            {/* Retry button */}
            <button
              id="error-boundary-retry-btn"
              onClick={this.handleReset}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-factory-500/20 border border-factory-500/30 text-factory-400 font-semibold text-sm hover:bg-factory-500/30 transition-all active:scale-95"
            >
              <RefreshCw className="w-4 h-4" />
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
