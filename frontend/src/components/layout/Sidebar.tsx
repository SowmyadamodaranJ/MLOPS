import { NavLink } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
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
  ChevronLeft,
  ChevronRight,
  Info,
  Sliders,
  X,
} from 'lucide-react';

interface SidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  isMobileOpen: boolean;
  onCloseMobile: () => void;
  onOpenSettings: () => void;
}

const mainNavItems = [
  { label: 'Command Center', path: '/', icon: LayoutDashboard },
  { label: 'Machine Fleet', path: '/machines', icon: Factory },
  { label: 'Failure Predictions', path: '/predictions', icon: Zap },
  { label: 'AI Explainability', path: '/explainability', icon: BrainCircuit },
  { label: 'Fleet Analytics', path: '/analytics', icon: BarChart3 },
];

const monitoringNavItems = [
  { label: 'Executive Monitoring', path: '/monitoring', icon: Activity },
  { label: 'System Health', path: '/system-health', icon: Gauge },
  { label: 'Model Monitoring', path: '/model-monitoring', icon: Layers },
  { label: 'Data & Model Drift', path: '/data-drift', icon: Activity },
  { label: 'Alerts Center', path: '/alerts', icon: Bell },
  { label: 'Prediction Audit Logs', path: '/prediction-logs', icon: ScrollText },
  { label: 'MLflow Platform', path: '/mlflow-dashboard', icon: GitBranch },
  { label: 'Model Registry', path: '/models', icon: FileText },
  { label: 'About Platform', path: '/about', icon: Info },
];

export default function Sidebar({
  isCollapsed,
  onToggleCollapse,
  isMobileOpen,
  onCloseMobile,
  onOpenSettings,
}: SidebarProps) {
  const sidebarWidth = isCollapsed ? 'w-20' : 'w-64';

  const sidebarContent = (
    <div className="h-full flex flex-col justify-between">
      {/* Brand Header */}
      <div>
        <div className={`flex items-center justify-between px-5 py-5 border-b border-white/[0.06] ${isCollapsed ? 'justify-center px-2' : ''}`}>
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-factory-500 via-purple-600 to-cyan-500 shadow-lg shadow-factory-500/20 flex-shrink-0">
              <Factory className="w-5 h-5 text-white" />
            </div>
            {!isCollapsed && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="whitespace-nowrap"
              >
                <h1 className="text-sm font-bold text-white tracking-tight">Smart Factory AI</h1>
                <p className="text-[10px] text-cyan-400 font-semibold tracking-wide">ENTERPRISE PLATFORM</p>
              </motion.div>
            )}
          </div>

          {/* Toggle Button for desktop */}
          <button
            onClick={onToggleCollapse}
            className="hidden lg:flex items-center justify-center w-7 h-7 rounded-lg bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] text-gray-400 hover:text-white transition-colors"
          >
            {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Navigation Items */}
        <nav className="px-3 py-4 space-y-6 overflow-y-auto max-h-[calc(100vh-180px)] custom-scrollbar">
          {/* Core Navigation */}
          <div>
            {!isCollapsed && (
              <p className="px-3 mb-2 text-[10px] font-bold text-gray-500 uppercase tracking-[0.15em]">
                Core Operations
              </p>
            )}
            <div className="space-y-1">
              {mainNavItems.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === '/'}
                  onClick={onCloseMobile}
                  className={({ isActive }) =>
                    `relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all duration-200 cursor-pointer ${
                      isCollapsed ? 'justify-center px-0' : ''
                    } ${
                      isActive
                        ? 'bg-factory-500/10 text-factory-400 border border-factory-500/15 shadow-sm shadow-factory-500/10'
                        : 'text-gray-400 hover:bg-white/[0.04] hover:text-gray-200'
                    }`
                  }
                  title={isCollapsed ? item.label : undefined}
                >
                  {({ isActive }) => (
                    <>
                      <item.icon
                        className={`w-4 h-4 flex-shrink-0 ${
                          isActive ? 'text-factory-400' : 'text-gray-400'
                        }`}
                      />
                      {!isCollapsed && <span>{item.label}</span>}
                      {isActive && (
                        <motion.div
                          layoutId="sidebar-indicator"
                          className="absolute left-0 w-[3px] h-5 rounded-r-full bg-factory-500"
                          transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                        />
                      )}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          </div>

          {/* Monitoring Navigation */}
          <div>
            {!isCollapsed && (
              <p className="px-3 mb-2 text-[10px] font-bold text-cyan-500 uppercase tracking-[0.15em]">
                Observability & Drift
              </p>
            )}
            <div className="space-y-1">
              {monitoringNavItems.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  onClick={onCloseMobile}
                  className={({ isActive }) =>
                    `relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all duration-200 cursor-pointer ${
                      isCollapsed ? 'justify-center px-0' : ''
                    } ${
                      isActive
                        ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/15 shadow-sm shadow-cyan-500/10'
                        : 'text-gray-400 hover:bg-white/[0.04] hover:text-gray-200'
                    }`
                  }
                  title={isCollapsed ? item.label : undefined}
                >
                  {({ isActive }) => (
                    <>
                      <item.icon
                        className={`w-4 h-4 flex-shrink-0 ${
                          isActive ? 'text-cyan-400' : 'text-gray-400'
                        }`}
                      />
                      {!isCollapsed && <span>{item.label}</span>}
                      {isActive && (
                        <motion.div
                          layoutId="sidebar-indicator"
                          className="absolute left-0 w-[3px] h-5 rounded-r-full bg-cyan-400"
                          transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                        />
                      )}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          </div>
        </nav>
      </div>

      {/* Footer / Settings Quick Trigger */}
      <div className="p-3 border-t border-white/[0.06]">
        <button
          onClick={onOpenSettings}
          className={`w-full p-2.5 rounded-xl bg-white/[0.03] border border-white/[0.06] hover:bg-white/[0.06] flex items-center gap-3 text-xs font-semibold text-gray-300 transition-colors ${
            isCollapsed ? 'justify-center px-0' : ''
          }`}
          title={isCollapsed ? 'Settings' : undefined}
        >
          <Sliders className="w-4 h-4 text-factory-400 flex-shrink-0" />
          {!isCollapsed && <span>Platform Settings</span>}
        </button>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Fixed Sidebar */}
      <motion.aside
        animate={{ width: isCollapsed ? 80 : 256 }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        className="hidden lg:flex fixed left-0 top-0 z-40 h-screen border-r border-white/[0.06] bg-[#0c0d14]/95 backdrop-blur-xl flex-col"
      >
        {sidebarContent}
      </motion.aside>

      {/* Mobile Drawer */}
      <AnimatePresence>
        {isMobileOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onCloseMobile}
              className="lg:hidden fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
            />
            <motion.aside
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
              className="lg:hidden fixed left-0 top-0 z-50 h-screen w-64 border-r border-white/[0.06] bg-[#0c0d14] flex flex-col"
            >
              <div className="flex justify-end p-3">
                <button
                  onClick={onCloseMobile}
                  className="p-1 rounded-lg hover:bg-white/10 text-gray-400 hover:text-white"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              {sidebarContent}
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
