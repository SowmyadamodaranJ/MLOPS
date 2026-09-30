import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Settings,
  Palette,
  Bell,
  Cpu,
  Check,
  Server,
  Layers,
  Database,
  RefreshCw,
  Sliders,
} from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onReplaySplash?: () => void;
}

const THEMES = [
  { id: 'dark', label: 'Enterprise Dark', color: '#07080b', border: '#6366f1' },
  { id: 'obsidian', label: 'Obsidian Black', color: '#030305', border: '#8b5cf6' },
  { id: 'space', label: 'Deep Space Blue', color: '#0a0d18', border: '#06b6d4' },
  { id: 'cyber', label: 'Cyberpunk Neon', color: '#0c0814', border: '#ec4899' },
];

const ACCENT_COLORS = [
  { id: 'indigo', label: 'Indigo', bg: 'bg-indigo-500', hex: '#6366f1' },
  { id: 'cyan', label: 'Cyan', bg: 'bg-cyan-500', hex: '#06b6d4' },
  { id: 'emerald', label: 'Emerald', bg: 'bg-emerald-500', hex: '#10b981' },
  { id: 'violet', label: 'Violet', bg: 'bg-purple-500', hex: '#a855f7' },
];

export default function SettingsModal({ isOpen, onClose, onReplaySplash }: SettingsModalProps) {
  const [activeTheme, setActiveTheme] = useState('dark');
  const [activeAccent, setActiveAccent] = useState('indigo');

  // Notification toggles
  const [notifyAlerts, setNotifyAlerts] = useState(true);
  const [notifyFailures, setNotifyFailures] = useState(true);
  const [notifyMlflow, setNotifyMlflow] = useState(false);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="fixed inset-0 bg-black/70 backdrop-blur-md"
        />

        {/* Dialog Window */}
        <motion.div
          initial={{ scale: 0.95, opacity: 0, y: 10 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.95, opacity: 0, y: 10 }}
          className="relative z-10 w-full max-w-lg glass-card bg-[#0b0c13]/95 border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.08] bg-white/[0.02]">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-factory-500/10 text-factory-400">
                <Settings className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white">Platform Settings</h3>
                <p className="text-[11px] text-gray-500">Customization, notification triggers & system telemetry</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] text-gray-400 hover:text-white flex items-center justify-center transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Body */}
          <div className="p-6 space-y-6 overflow-y-auto custom-scrollbar flex-1">
            {/* Theme Selector */}
            <div className="space-y-3">
              <label className="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                <Palette className="w-4 h-4 text-factory-400" /> UI Theme Preset
              </label>
              <div className="grid grid-cols-2 gap-3">
                {THEMES.map((t) => (
                  <button
                    key={t.id}
                    onClick={() => setActiveTheme(t.id)}
                    className={`p-3 rounded-xl border text-left flex items-center justify-between transition-all ${
                      activeTheme === t.id
                        ? 'bg-white/[0.06] border-factory-500/50 shadow-lg shadow-factory-500/10'
                        : 'bg-white/[0.02] border-white/[0.06] hover:bg-white/[0.04]'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <span className="w-4 h-4 rounded-full border border-white/20" style={{ backgroundColor: t.color }} />
                      <span className="text-xs font-semibold text-gray-200">{t.label}</span>
                    </div>
                    {activeTheme === t.id && <Check className="w-4 h-4 text-factory-400" />}
                  </button>
                ))}
              </div>
            </div>

            {/* Accent Color Customizer */}
            <div className="space-y-3">
              <label className="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                <Sliders className="w-4 h-4 text-cyan-400" /> Accent Highlight
              </label>
              <div className="flex items-center gap-3">
                {ACCENT_COLORS.map((a) => (
                  <button
                    key={a.id}
                    onClick={() => setActiveAccent(a.id)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-xl border text-xs font-semibold transition-all ${
                      activeAccent === a.id
                        ? 'bg-white/[0.08] border-white/20 text-white'
                        : 'bg-white/[0.02] border-white/[0.06] text-gray-400 hover:text-gray-200'
                    }`}
                  >
                    <span className={`w-3 h-3 rounded-full ${a.bg}`} />
                    <span>{a.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Notification Preferences */}
            <div className="space-y-3">
              <label className="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                <Bell className="w-4 h-4 text-amber-400" /> Notification Triggers
              </label>
              <div className="space-y-2 glass-card p-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-300">Critical Failure Alerts</span>
                  <input
                    type="checkbox"
                    checked={notifyAlerts}
                    onChange={(e) => setNotifyAlerts(e.target.checked)}
                    className="accent-factory-500 w-4 h-4 rounded cursor-pointer"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-300">Prediction Logs Digest</span>
                  <input
                    type="checkbox"
                    checked={notifyFailures}
                    onChange={(e) => setNotifyFailures(e.target.checked)}
                    className="accent-factory-500 w-4 h-4 rounded cursor-pointer"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-300">MLflow Model Registry Events</span>
                  <input
                    type="checkbox"
                    checked={notifyMlflow}
                    onChange={(e) => setNotifyMlflow(e.target.checked)}
                    className="accent-factory-500 w-4 h-4 rounded cursor-pointer"
                  />
                </div>
              </div>
            </div>

            {/* System Info Matrix */}
            <div className="space-y-3">
              <label className="text-xs font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                <Server className="w-4 h-4 text-emerald-400" /> System Runtime Environment
              </label>
              <div className="glass-card p-4 text-xs space-y-2 text-gray-300">
                <div className="flex justify-between border-b border-white/[0.04] pb-1.5">
                  <span className="text-gray-500">Platform Version</span>
                  <span className="font-bold text-white">v2.0.0 (Enterprise)</span>
                </div>
                <div className="flex justify-between border-b border-white/[0.04] pb-1.5">
                  <span className="text-gray-500">API Runtime</span>
                  <span className="font-bold text-emerald-400">Flask + Gunicorn 21.2</span>
                </div>
                <div className="flex justify-between border-b border-white/[0.04] pb-1.5">
                  <span className="text-gray-500">Active ML Model</span>
                  <span className="font-bold text-cyan-400">XGBoost Classifier</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Tracking Server</span>
                  <span className="font-bold text-purple-400">MLflow v2.10 (SQLite)</span>
                </div>
              </div>
            </div>

            {/* Replay Splash Screen */}
            {onReplaySplash && (
              <div className="pt-2">
                <button
                  onClick={() => {
                    onClose();
                    onReplaySplash();
                  }}
                  className="w-full h-10 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] text-xs font-semibold text-gray-300 flex items-center justify-center gap-2 transition-all"
                >
                  <RefreshCw className="w-3.5 h-3.5 text-factory-400" /> Replay AI Startup Sequence
                </button>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-white/[0.08] bg-[#0c0d14] flex justify-end">
            <button
              onClick={onClose}
              className="px-5 py-2 rounded-xl bg-gradient-to-r from-factory-500 to-purple-600 text-white text-xs font-bold shadow-lg shadow-factory-500/20 hover:opacity-90 transition-all"
            >
              Save & Apply
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
