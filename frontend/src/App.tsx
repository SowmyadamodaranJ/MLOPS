import { useState, useEffect } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';

// Layout Components
import Sidebar from './components/layout/Sidebar';
import TopNavbar from './components/layout/TopNavbar';
import Footer from './components/layout/Footer';
import SplashScreen from './components/layout/SplashScreen';
import CommandPalette from './components/layout/CommandPalette';
import SettingsModal from './components/ui/SettingsModal';
import MachineDetailPanel, { MachineDetail } from './components/ui/MachineDetailPanel';
import { fetchMachineDecision } from './services/api';

// Pages
import Dashboard from './pages/Dashboard';
import MachinesPage from './pages/MachinesPage';
import Predictions from './pages/Predictions';
import Analytics from './pages/Analytics';
import Explainability from './pages/Explainability';
import Models from './pages/Models';
import SystemStatus from './pages/SystemStatus';
import SystemHealthPage from './pages/SystemHealthPage';
import ModelMonitoringPage from './pages/ModelMonitoringPage';
import DataDriftPage from './pages/DataDriftPage';
import AlertsPage from './pages/AlertsPage';
import PredictionLogsPage from './pages/PredictionLogsPage';
import MlflowDashboardPage from './pages/MlflowDashboardPage';
import AboutPage from './pages/AboutPage';

export default function App() {
  const location = useLocation();

  // Splash Screen State
  const [showSplash, setShowSplash] = useState(() => {
    return !sessionStorage.getItem('pdm_splash_shown');
  });

  // Layout & Modal States
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const [isCommandOpen, setIsCommandOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [selectedMachine, setSelectedMachine] = useState<MachineDetail | null>(null);

  const handleSplashComplete = () => {
    sessionStorage.setItem('pdm_splash_shown', 'true');
    setShowSplash(false);
  };

  const handleReplaySplash = () => {
    setShowSplash(true);
  };

  const handleSelectMachineId = async (mId: string) => {
    try {
      const idNum = parseInt(mId, 10);
      const res = await fetchMachineDecision(idNum);
      const d = res.data || res;
      const feats = d.features || {};
      const statusMap: Record<string, 'Healthy' | 'Warning' | 'Critical'> = {
        'Critical': 'Critical',
        'CRITICAL': 'Critical',
        'High': 'Critical',
        'Warning': 'Warning',
        'WARNING': 'Warning',
        'Medium': 'Warning',
        'Healthy': 'Healthy',
        'HEALTHY': 'Healthy',
        'MONITOR': 'Healthy',
        'Low': 'Healthy'
      };
      const status = statusMap[d.risk_tier] || (d.prediction?.prediction === 1 ? 'Critical' : 'Healthy');
      const failProb = d.prediction?.probability ?? (status === 'Critical' ? 0.94 : 0.04);

      setSelectedMachine({
        machineID: d.machine_id || mId,
        model: feats.model || 'model3',
        age: feats.age || 10,
        status,
        healthScore: Math.round(d.health_score ?? (100 - failProb * 100)),
        failureProbability: failProb,
        confidence: d.confidence ?? 0.95,
        volt: Number(feats.volt ?? 170.0),
        rotate: Number(feats.rotate ?? 450.0),
        pressure: Number(feats.pressure ?? 100.0),
        vibration: Number(feats.vibration ?? 40.0),
        estDowntimeHrs: status === 'Critical' ? 48 : status === 'Warning' ? 12 : 0,
        estRepairCostUsd: status === 'Critical' ? 18500 : status === 'Warning' ? 4500 : 0,
        recommendation: d.recommendation || (status === 'Critical' ? 'Immediate maintenance inspection recommended.' : 'Telemetry parameters operating within normal bounds.'),
      });
    } catch {
      // Fallback if machine ID lookup fails
      setSelectedMachine({
        machineID: mId,
        model: 'model3',
        age: 10,
        status: 'Healthy',
        healthScore: 92,
        failureProbability: 0.045,
        confidence: 0.95,
        volt: 170.0,
        rotate: 450.0,
        pressure: 100.0,
        vibration: 40.0,
        estDowntimeHrs: 0,
        estRepairCostUsd: 0,
        recommendation: 'Telemetry parameters operating within normal bounds.',
      });
    }
  };

  return (
    <div className="min-h-screen bg-[#07080b] text-gray-200 flex font-sans overflow-x-hidden selection:bg-factory-500/30 selection:text-white">
      {/* Animated Splash Screen */}
      <AnimatePresence>
        {showSplash && <SplashScreen onComplete={handleSplashComplete} />}
      </AnimatePresence>

      {/* Sidebar Layout */}
      <Sidebar
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
        isMobileOpen={isMobileOpen}
        onCloseMobile={() => setIsMobileOpen(false)}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* Main Content Workspace */}
      <div
        className={`flex-1 flex flex-col min-h-screen transition-all duration-300 ${
          isSidebarCollapsed ? 'lg:pl-20' : 'lg:pl-64'
        }`}
      >
        {/* Top Header Navbar */}
        <TopNavbar
          onOpenCommandPalette={() => setIsCommandOpen(true)}
          onOpenSettings={() => setIsSettingsOpen(true)}
          onToggleMobileMenu={() => setIsMobileOpen(true)}
        />

        {/* Dynamic Animated Page Workspace */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-[1500px] w-full mx-auto flex flex-col justify-between">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25, ease: 'easeInOut' }}
            >
              <Routes location={location}>
                <Route path="/" element={<Dashboard />} />
                <Route path="/machines" element={<MachinesPage />} />
                <Route path="/predictions" element={<Predictions />} />
                <Route path="/explainability" element={<Explainability />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/models" element={<Models />} />
                <Route path="/monitoring" element={<SystemStatus />} />
                <Route path="/system-health" element={<SystemHealthPage />} />
                <Route path="/model-monitoring" element={<ModelMonitoringPage />} />
                <Route path="/data-drift" element={<DataDriftPage />} />
                <Route path="/alerts" element={<AlertsPage />} />
                <Route path="/prediction-logs" element={<PredictionLogsPage />} />
                <Route path="/mlflow-dashboard" element={<MlflowDashboardPage />} />
                <Route path="/about" element={<AboutPage />} />
              </Routes>
            </motion.div>
          </AnimatePresence>

          {/* Footer */}
          <Footer />
        </main>
      </div>

      {/* Global Command Palette Modal */}
      <CommandPalette
        isOpen={isCommandOpen}
        onClose={() => setIsCommandOpen(false)}
        onSelectMachine={handleSelectMachineId}
      />

      {/* Settings & Preferences Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        onReplaySplash={handleReplaySplash}
      />

      {/* Global Machine Side Panel Drawer */}
      <MachineDetailPanel
        machine={selectedMachine}
        isOpen={!!selectedMachine}
        onClose={() => setSelectedMachine(null)}
      />
    </div>
  );
}
