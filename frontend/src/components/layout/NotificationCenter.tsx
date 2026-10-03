import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bell,
  CheckCircle2,
  AlertTriangle,
  Zap,
  Layers,
  GitBranch,
  X,
  Trash2,
  CheckCheck,
} from 'lucide-react';

export interface NotificationItem {
  id: string;
  type: 'alert' | 'prediction' | 'model' | 'mlflow' | 'system';
  title: string;
  message: string;
  time: string;
  read: boolean;
}

const INITIAL_NOTIFICATIONS: NotificationItem[] = [
  {
    id: 'n1',
    type: 'alert',
    title: 'High Failure Risk Detected',
    message: 'Machine M-094 error rate elevated. Risk score 99.9% (Critical).',
    time: '2 mins ago',
    read: false,
  },
  {
    id: 'n2',
    type: 'prediction',
    title: 'Diagnostic Prediction Completed',
    message: 'Machine M-035 processed with XGBoost champion model.',
    time: '14 mins ago',
    read: false,
  },
  {
    id: 'n3',
    type: 'model',
    title: 'Model Version Updated',
    message: 'XGBoost v2.0.0 pipeline registered as active champion model.',
    time: '1 hour ago',
    read: false,
  },
  {
    id: 'n4',
    type: 'mlflow',
    title: 'MLflow Run Logged',
    message: 'Experiment "predictive-maintenance" logged authentic F1 0.825 & ROC-AUC 0.887.',
    time: '3 hours ago',
    read: true,
  },
  {
    id: 'n5',
    type: 'system',
    title: 'Maintenance Schedule Confirmed',
    message: 'Inspection scheduled for Machine M-094 within immediate P1 window.',
    time: '5 hours ago',
    read: true,
  },
];

interface NotificationCenterProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function NotificationCenter({ isOpen, onClose }: NotificationCenterProps) {
  const [notifications, setNotifications] = useState<NotificationItem[]>(INITIAL_NOTIFICATIONS);
  const [activeFilter, setActiveFilter] = useState<string>('all');

  const unreadCount = notifications.filter((n) => !n.read).length;

  const markAllAsRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  };

  const clearAll = () => {
    setNotifications([]);
  };

  const dismiss = (id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  const filtered = notifications.filter((n) => {
    if (activeFilter === 'all') return true;
    return n.type === activeFilter;
  });

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="absolute right-0 top-12 z-50 w-80 sm:w-96 glass-card bg-[#0b0c13]/95 border border-white/10 rounded-2xl shadow-2xl overflow-hidden backdrop-blur-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.08] bg-white/[0.02]">
          <div className="flex items-center gap-2">
            <Bell className="w-4 h-4 text-factory-400" />
            <h3 className="text-sm font-bold text-white tracking-tight">Notifications</h3>
            {unreadCount > 0 && (
              <span className="text-[10px] font-extrabold px-2 py-0.5 rounded-full bg-rose-500 text-white">
                {unreadCount} New
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            {unreadCount > 0 && (
              <button
                onClick={markAllAsRead}
                title="Mark all as read"
                className="p-1.5 rounded-lg hover:bg-white/[0.06] text-gray-400 hover:text-white transition-colors"
              >
                <CheckCheck className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={clearAll}
              title="Clear notifications"
              className="p-1.5 rounded-lg hover:bg-white/[0.06] text-gray-400 hover:text-rose-400 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-white/[0.06] text-gray-400 hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-1 px-4 py-2 border-b border-white/[0.04] overflow-x-auto text-[11px]">
          {['all', 'alert', 'prediction', 'model', 'mlflow'].map((f) => (
            <button
              key={f}
              onClick={() => setActiveFilter(f)}
              className={`px-2.5 py-1 rounded-lg font-semibold uppercase tracking-wider transition-colors ${
                activeFilter === f
                  ? 'bg-factory-500/20 text-factory-400 border border-factory-500/30'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              {f}
            </button>
          ))}
        </div>

        {/* List */}
        <div className="max-h-80 overflow-y-auto divide-y divide-white/[0.04] custom-scrollbar">
          {filtered.length === 0 ? (
            <div className="py-10 text-center text-xs text-gray-500 font-medium">
              No notifications available.
            </div>
          ) : (
            filtered.map((item) => {
              const iconMap = {
                alert: <AlertTriangle className="w-4 h-4 text-rose-400" />,
                prediction: <Zap className="w-4 h-4 text-cyan-400" />,
                model: <Layers className="w-4 h-4 text-purple-400" />,
                mlflow: <GitBranch className="w-4 h-4 text-indigo-400" />,
                system: <CheckCircle2 className="w-4 h-4 text-emerald-400" />,
              };

              return (
                <div
                  key={item.id}
                  className={`p-4 flex items-start justify-between gap-3 transition-colors ${
                    item.read ? 'bg-transparent opacity-75' : 'bg-white/[0.02]'
                  } hover:bg-white/[0.04] group`}
                >
                  <div className="flex items-start gap-3">
                    <div className="p-2 rounded-xl bg-white/[0.04] border border-white/[0.08] mt-0.5">
                      {iconMap[item.type]}
                    </div>
                    <div className="space-y-0.5">
                      <p className="text-xs font-bold text-white flex items-center gap-1.5">
                        {item.title}
                        {!item.read && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />}
                      </p>
                      <p className="text-[11px] text-gray-400 leading-snug">{item.message}</p>
                      <span className="text-[9px] text-gray-600 font-medium block pt-1">{item.time}</span>
                    </div>
                  </div>

                  <button
                    onClick={() => dismiss(item.id)}
                    className="opacity-0 group-hover:opacity-100 p-1 text-gray-500 hover:text-rose-400 transition-opacity"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              );
            })
          )}
        </div>
      </div>
    </AnimatePresence>
  );
}
