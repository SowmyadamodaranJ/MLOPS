import { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { Bell, Search, User, Menu, Sliders, Sparkles } from 'lucide-react';
import NotificationCenter from './NotificationCenter';

interface TopNavbarProps {
  onOpenCommandPalette: () => void;
  onOpenSettings: () => void;
  onToggleMobileMenu: () => void;
}

const pageTitles: Record<string, { title: string; subtitle: string }> = {
  '/': { title: 'Factory Command Center', subtitle: 'Real-time fleet health & predictive AI insights' },
  '/machines': { title: 'Machine Fleet Management', subtitle: 'Telemetry parameters, status & maintenance scheduling' },
  '/predictions': { title: 'Failure Predictions', subtitle: 'Simulate sensor tuning & execute diagnostic model inference' },
  '/analytics': { title: 'Executive Fleet Analytics', subtitle: 'Aggregated telemetry performance & downtime distribution' },
  '/explainability': { title: 'AI Explainability (SHAP)', subtitle: 'Local & global feature contribution explanations' },
  '/monitoring': { title: 'Executive System Monitoring', subtitle: 'Overall health score & real-time telemetry matrix' },
  '/system-health': { title: 'System Health Observability', subtitle: 'Backend, DB connection, and artifact status telemetry' },
  '/model-monitoring': { title: 'Model Performance & Drift', subtitle: 'Inference latency, confidence, & probability distribution' },
  '/data-drift': { title: 'Data & Model Drift', subtitle: 'Evidently AI statistical drift report' },
  '/alerts': { title: 'Alerts & Critical Feed', subtitle: 'Active system warnings & critical failure logs' },
  '/prediction-logs': { title: 'Prediction Audit Logs', subtitle: 'Historical inference logs with CSV export' },
  '/mlflow-dashboard': { title: 'MLflow Platform', subtitle: 'Experiment tracking, parameters & metric logs' },
  '/models': { title: 'Model Registry', subtitle: 'Champion ML model artifact metadata' },
  '/about': { title: 'About Platform', subtitle: 'Industry 4.0 architecture, ML pipeline & developer specs' },
};

export default function TopNavbar({
  onOpenCommandPalette,
  onOpenSettings,
  onToggleMobileMenu,
}: TopNavbarProps) {
  const location = useLocation();
  const [isNotifOpen, setIsNotifOpen] = useState(false);

  const pageInfo = pageTitles[location.pathname] || {
    title: 'Command Center',
    subtitle: 'Industry 4.0 Platform',
  };

  return (
    <header className="sticky top-0 z-30 h-16 border-b border-white/[0.06] bg-[#0c0d14]/80 backdrop-blur-xl flex items-center justify-between px-4 sm:px-8">
      {/* Left: Mobile Menu Toggle & Title */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleMobileMenu}
          className="lg:hidden p-2 rounded-xl bg-white/[0.04] border border-white/[0.08] text-gray-400 hover:text-white"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div>
          <h2 className="text-base sm:text-lg font-bold text-white tracking-tight flex items-center gap-2">
            {pageInfo.title}
          </h2>
          <p className="text-[10px] sm:text-[11px] text-gray-500 -mt-0.5 font-medium">
            {pageInfo.subtitle}
          </p>
        </div>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-2 sm:gap-3">
        {/* Command Palette Trigger */}
        <button
          onClick={onOpenCommandPalette}
          className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] text-xs font-semibold text-gray-400 hover:text-gray-200 transition-all cursor-pointer"
        >
          <Search className="w-3.5 h-3.5 text-gray-400" />
          <span className="hidden sm:inline">Search...</span>
          <kbd className="hidden sm:inline-block px-1.5 py-0.5 text-[9px] font-bold text-gray-400 bg-white/10 rounded">
            Ctrl + K
          </kbd>
        </button>

        {/* Settings Modal Trigger */}
        <button
          onClick={onOpenSettings}
          title="Platform Settings"
          className="p-2 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] text-gray-400 hover:text-white transition-colors"
        >
          <Sliders className="w-4 h-4" />
        </button>

        {/* Notifications Dropdown */}
        <div className="relative">
          <button
            onClick={() => setIsNotifOpen(!isNotifOpen)}
            title="Notification Center"
            className="relative p-2 rounded-xl bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] text-gray-400 hover:text-white transition-colors"
          >
            <Bell className="w-4 h-4" />
            <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-rose-500 text-[9px] font-extrabold text-white flex items-center justify-center shadow-lg shadow-rose-500/50">
              3
            </span>
          </button>

          <NotificationCenter isOpen={isNotifOpen} onClose={() => setIsNotifOpen(false)} />
        </div>

        {/* User Badge */}
        <div className="flex items-center gap-2.5 pl-2 sm:pl-3 border-l border-white/[0.08]">
          <div className="text-right hidden md:block">
            <p className="text-xs font-bold text-gray-200">Admin Engineer</p>
            <span className="text-[9px] font-semibold text-emerald-400 uppercase tracking-wider block">
              Plant Operator
            </span>
          </div>
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-factory-500 to-purple-600 flex items-center justify-center text-white shadow-lg shadow-factory-500/20">
            <User className="w-4 h-4" />
          </div>
        </div>
      </div>
    </header>
  );
}
