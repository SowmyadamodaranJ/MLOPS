import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  LayoutDashboard,
  Zap,
  BrainCircuit,
  Activity,
  Gauge,
  Bell,
  ScrollText,
  GitBranch,
  FileText,
  BarChart3,
  Factory,
  Layers,
  Info,
  Wrench,
  Download,
  Sliders,
  ArrowRight,
  X,
} from 'lucide-react';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectMachine?: (machineId: string) => void;
}

export default function CommandPalette({ isOpen, onClose, onSelectMachine }: CommandPaletteProps) {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');

  // Keyboard shortcut listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if (isOpen) onClose();
        else setQuery('');
      } else if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const COMMANDS = [
    // Navigation
    { id: 'nav-dash', title: 'Factory Command Center', type: 'Page Navigation', icon: LayoutDashboard, action: () => navigate('/') },
    { id: 'nav-machines', title: 'Machine Fleet Management', type: 'Page Navigation', icon: Factory, action: () => navigate('/machines') },
    { id: 'nav-predict', title: 'Real-Time Failure Predictions', type: 'Page Navigation', icon: Zap, action: () => navigate('/predictions') },
    { id: 'nav-shap', title: 'AI Explainability & SHAP', type: 'Page Navigation', icon: BrainCircuit, action: () => navigate('/explainability') },
    { id: 'nav-analytics', title: 'Executive Fleet Analytics', type: 'Page Navigation', icon: BarChart3, action: () => navigate('/analytics') },
    { id: 'nav-monitoring', title: 'Executive System Monitoring', type: 'Page Navigation', icon: Activity, action: () => navigate('/monitoring') },
    { id: 'nav-health', title: 'System Health Observability', type: 'Page Navigation', icon: Gauge, action: () => navigate('/system-health') },
    { id: 'nav-model-mon', title: 'Model Monitoring & Stability', type: 'Page Navigation', icon: Layers, action: () => navigate('/model-monitoring') },
    { id: 'nav-drift', title: 'Data & Model Drift (Evidently AI)', type: 'Page Navigation', icon: Activity, action: () => navigate('/data-drift') },
    { id: 'nav-alerts', title: 'Alerts & Critical Feed', type: 'Page Navigation', icon: Bell, action: () => navigate('/alerts') },
    { id: 'nav-audit', title: 'Prediction Audit Logs', type: 'Page Navigation', icon: ScrollText, action: () => navigate('/prediction-logs') },
    { id: 'nav-mlflow', title: 'MLflow Tracking Platform', type: 'Page Navigation', icon: GitBranch, action: () => navigate('/mlflow-dashboard') },
    { id: 'nav-models', title: 'Model Registry', type: 'Page Navigation', icon: FileText, action: () => navigate('/models') },
    { id: 'nav-about', title: 'About Platform & Architecture', type: 'Page Navigation', icon: Info, action: () => navigate('/about') },

    // Machines
    ...Array.from({ length: 15 }, (_, i) => {
      const mId = (i + 1).toString();
      return {
        id: `machine-${mId}`,
        title: `Inspect Machine M-${mId}`,
        type: 'Machine Fleet',
        icon: CpuIcon,
        action: () => {
          onSelectMachine?.(mId);
          onClose();
        },
      };
    }),
  ];

  function CpuIcon(props: any) {
    return <Factory {...props} />;
  }

  const filtered = COMMANDS.filter((cmd) =>
    cmd.title.toLowerCase().includes(query.toLowerCase()) ||
    cmd.type.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 p-4">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="fixed inset-0 bg-black/70 backdrop-blur-md"
        />

        {/* Command Modal */}
        <motion.div
          initial={{ scale: 0.95, opacity: 0, y: -10 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.95, opacity: 0, y: -10 }}
          className="relative z-10 w-full max-w-xl glass-card bg-[#0b0c13]/95 border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col"
        >
          {/* Input Bar */}
          <div className="flex items-center px-4 py-3.5 border-b border-white/[0.08] bg-white/[0.02]">
            <Search className="w-5 h-5 text-gray-400 mr-3 flex-shrink-0" />
            <input
              type="text"
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Type a command, search machines, or navigate..."
              className="w-full bg-transparent text-sm text-white placeholder-gray-500 focus:outline-none"
            />
            <button
              onClick={onClose}
              className="p-1 rounded-lg hover:bg-white/[0.06] text-gray-500 hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Results List */}
          <div className="max-h-96 overflow-y-auto p-2 divide-y divide-white/[0.04] custom-scrollbar">
            {filtered.length === 0 ? (
              <div className="py-12 text-center text-xs text-gray-500 font-medium">
                No matching commands or machines found.
              </div>
            ) : (
              filtered.map((item) => {
                const IconComponent = item.icon;
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      item.action();
                      onClose();
                    }}
                    className="w-full p-3 rounded-xl flex items-center justify-between hover:bg-white/[0.06] transition-all group text-left"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-white/[0.04] border border-white/[0.08] text-gray-400 group-hover:text-cyan-400 group-hover:border-cyan-500/30 transition-colors">
                        <IconComponent className="w-4 h-4" />
                      </div>
                      <div>
                        <p className="text-xs font-bold text-gray-200 group-hover:text-white transition-colors">
                          {item.title}
                        </p>
                        <span className="text-[10px] text-gray-500 font-medium uppercase tracking-wider">
                          {item.type}
                        </span>
                      </div>
                    </div>
                    <ArrowRight className="w-4 h-4 text-gray-600 opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
                  </button>
                );
              })
            )}
          </div>

          {/* Footer Shortcuts */}
          <div className="px-4 py-2.5 border-t border-white/[0.08] bg-[#0c0d14] flex items-center justify-between text-[10px] text-gray-500 font-semibold uppercase tracking-wider">
            <div className="flex items-center gap-3">
              <span><kbd className="px-1.5 py-0.5 rounded bg-white/10 text-gray-300">↑</kbd> <kbd className="px-1.5 py-0.5 rounded bg-white/10 text-gray-300">↓</kbd> Navigate</span>
              <span><kbd className="px-1.5 py-0.5 rounded bg-white/10 text-gray-300">↵</kbd> Select</span>
              <span><kbd className="px-1.5 py-0.5 rounded bg-white/10 text-gray-300">ESC</kbd> Close</span>
            </div>
            <span className="text-cyan-400">Ctrl + K</span>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
