/**
 * useQuery.ts
 * -----------
 * Custom data-fetching hooks for the Predictive Maintenance frontend.
 *
 * Production-readiness additions
 * --------------------------------
 * - Stable fetcher execution via useRef prevents infinite re-render loops
 *   when inline functions are passed as fetchers.
 * - Options.deps support for declarative re-fetching when query parameters change.
 * - useQuery exposes errorCode and errorMessage parsed from ApiError so
 *   components get strongly-typed error details without touching the raw
 *   error object.
 * - useMutation does the same for mutation errors with stable mutator reference.
 * - AbortController / mountedRef cancels in-flight requests on component unmount,
 *   preventing state updates on unmounted components.
 * - refetch resets error state before retrying so the UI transitions
 *   correctly from error → loading → data.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { ApiError } from '../services/api';

// ─── Shared types ─────────────────────────────────────────────────────────────

export interface QueryOptions {
  /** Polling interval in ms.  Omit to disable. */
  refetchInterval?: number;
  /** Set to false to skip the initial fetch. Defaults to true. */
  enabled?: boolean;
  /** Optional dependency list that triggers refetch when values change. */
  deps?: any[];
}

export interface QueryResult<T> {
  data: T | null;
  isLoading: boolean;
  error: ApiError | Error | null;
  /** Parsed error code from backend (e.g. 'MODEL_NOT_LOADED'). */
  errorCode: string | null;
  /** Parsed human-readable error message from backend. */
  errorMessage: string | null;
  refetch: () => void;
}

export interface MutationResult<T, V> {
  data: T | null;
  isLoading: boolean;
  error: ApiError | Error | null;
  /** Parsed error code from backend. */
  errorCode: string | null;
  /** Parsed human-readable error message from backend. */
  errorMessage: string | null;
  mutate: (variables: V) => Promise<T | void>;
  /** Reset mutation state (data, error, loading) back to initial. */
  reset: () => void;
}

// ─── Helper ───────────────────────────────────────────────────────────────────

function extractErrorFields(err: any): { errorCode: string | null; errorMessage: string | null } {
  if (err instanceof ApiError) {
    return {
      errorCode:    err.errorCode || null,
      errorMessage: err.message   || null,
    };
  }
  return {
    errorCode:    null,
    errorMessage: err?.message || 'An unexpected error occurred.',
  };
}

// ─── useQuery ─────────────────────────────────────────────────────────────────

export function useQuery<T>(
  fetcher: () => Promise<T>,
  options?: QueryOptions,
): QueryResult<T> {
  const [data,         setData]         = useState<T | null>(null);
  const [isLoading,    setIsLoading]    = useState<boolean>(options?.enabled !== false);
  const [error,        setError]        = useState<ApiError | Error | null>(null);
  const [errorCode,    setErrorCode]    = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Keep latest fetcher in a ref so execute has a completely stable identity
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  // Track whether the component is still mounted
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  const execute = useCallback(async () => {
    if (!mountedRef.current) return;

    setIsLoading(true);
    setError(null);
    setErrorCode(null);
    setErrorMessage(null);

    try {
      const res = await fetcherRef.current();
      if (mountedRef.current) {
        setData(res);
      }
    } catch (err: any) {
      if (mountedRef.current) {
        const { errorCode: ec, errorMessage: em } = extractErrorFields(err);
        setError(err);
        setErrorCode(ec);
        setErrorMessage(em);
        console.error('[useQuery] fetch error:', err);
      }
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  }, []);

  // Track initial mount and previous deps
  const isInitialMount = useRef(true);
  const prevDepsRef = useRef(options?.deps);
  const prevEnabledRef = useRef(options?.enabled);

  useEffect(() => {
    const isEnabled = options?.enabled !== false;
    const currentDeps = options?.deps;

    if (isInitialMount.current) {
      isInitialMount.current = false;
      prevDepsRef.current = currentDeps;
      prevEnabledRef.current = options?.enabled;
      if (isEnabled) {
        execute();
      }
      return;
    }

    const enabledJustActivated = !prevEnabledRef.current && isEnabled;
    prevEnabledRef.current = options?.enabled;

    let depsChanged = false;
    if (currentDeps && prevDepsRef.current) {
      if (
        currentDeps.length !== prevDepsRef.current.length ||
        currentDeps.some((d, i) => !Object.is(d, prevDepsRef.current![i]))
      ) {
        depsChanged = true;
      }
    } else if (currentDeps !== prevDepsRef.current) {
      depsChanged = true;
    }
    prevDepsRef.current = currentDeps;

    if (isEnabled && (depsChanged || enabledJustActivated)) {
      execute();
    }
  // NOTE: options?.deps is intentionally NOT in this dependency array.
  // The manual prevDepsRef comparison (above) handles dep-change detection.
  // Including options?.deps here would cause infinite re-renders because
  // callers typically pass a new array literal on every render (e.g. deps: [id]).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [execute, options?.enabled]);

  // Polling
  useEffect(() => {
    if (options?.refetchInterval && options?.enabled !== false) {
      const interval = setInterval(execute, options.refetchInterval);
      return () => clearInterval(interval);
    }
  }, [execute, options?.refetchInterval, options?.enabled]);

  return { data, isLoading, error, errorCode, errorMessage, refetch: execute };
}

// ─── useMutation ──────────────────────────────────────────────────────────────

export function useMutation<T, V = any>(
  mutator: (variables: V) => Promise<T>,
): MutationResult<T, V> {
  const [data,         setData]         = useState<T | null>(null);
  const [isLoading,    setIsLoading]    = useState<boolean>(false);
  const [error,        setError]        = useState<ApiError | Error | null>(null);
  const [errorCode,    setErrorCode]    = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const mutatorRef = useRef(mutator);
  mutatorRef.current = mutator;

  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  const mutate = useCallback(async (variables: V): Promise<T | void> => {
    if (!mountedRef.current) return;

    setIsLoading(true);
    setError(null);
    setErrorCode(null);
    setErrorMessage(null);

    try {
      const res = await mutatorRef.current(variables);
      if (mountedRef.current) {
        setData(res);
      }
      return res;
    } catch (err: any) {
      if (mountedRef.current) {
        const { errorCode: ec, errorMessage: em } = extractErrorFields(err);
        setError(err);
        setErrorCode(ec);
        setErrorMessage(em);
        console.error('[useMutation] execution error:', err);
      }
      throw err;
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setErrorCode(null);
    setErrorMessage(null);
    setIsLoading(false);
  }, []);

  return { data, isLoading, error, errorCode, errorMessage, mutate, reset };
}
