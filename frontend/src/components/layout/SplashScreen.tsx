import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Factory, Brain, Cpu, Database, Network, Activity, CheckCircle2, Sparkles } from 'lucide-react';

interface SplashScreenProps {
  onComplete: () => void;
}

const STEPS = [
  { text: 'Loading AI Engine...', icon: Brain },
  { text: 'Loading ML Models...', icon: Cpu },
  { text: 'Connecting MLflow...', icon: Database },
  { text: 'Connecting Prediction API...', icon: Network },
  { text: 'Loading Factory Status...', icon: Activity },
  { text: 'Ready.', icon: Sparkles },
];

export default function SplashScreen({ onComplete }: SplashScreenProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const totalSteps = STEPS.length;
    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < totalSteps - 1) {
          const next = prev + 1;
          setProgress(Math.round(((next + 1) / totalSteps) * 100));
          return next;
        }
        clearInterval(interval);
        setTimeout(onComplete, 800);
        return prev;
      });
    }, 550);

    return () => clearInterval(interval);
  }, [onComplete]);

  return (
    <motion.div
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, scale: 1.05 }}
      transition={{ duration: 0.8, ease: 'easeInOut' }}
      className="fixed inset-0 z-50 bg-[#06070a] flex flex-col items-center justify-center p-6 overflow-hidden select-none"
    >
      {/* Radial Ambient Glow */}
      <div className="absolute w-[600px] h-[600px] rounded-full bg-gradient-to-tr from-factory-500/20 via-purple-600/20 to-cyan-500/10 blur-[120px] pointer-events-none animate-pulse" />

      {/* Main Container */}
      <div className="relative z-10 w-full max-w-sm flex flex-col items-center text-center space-y-8">
        {/* Brand Icon */}
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.6 }}
          className="relative flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-factory-500 via-purple-600 to-cyan-500 p-0.5 shadow-2xl shadow-factory-500/30"
        >
          <div className="w-full h-full rounded-[14px] bg-[#0c0d14] flex items-center justify-center">
            <Factory className="w-9 h-9 text-white animate-pulse" />
          </div>
        </motion.div>

        {/* Title */}
        <div className="space-y-1.5">
          <h1 className="text-xl font-extrabold text-white tracking-tight">Smart Factory AI</h1>
          <p className="text-xs font-semibold text-cyan-400 uppercase tracking-widest">
            INDUSTRY 4.0 COMMAND CENTER
          </p>
        </div>

        {/* Step Sequence Display */}
        <div className="w-full h-16 flex items-center justify-center glass-card px-6 bg-white/[0.02]">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentStep}
              initial={{ y: 10, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -10, opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="flex items-center gap-3 text-sm font-semibold text-gray-200"
            >
              {(() => {
                const IconComponent = STEPS[currentStep].icon;
                return <IconComponent className="w-5 h-5 text-factory-400 animate-spin-slow" />;
              })()}
              <span>{STEPS[currentStep].text}</span>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Progress Bar & Percentage */}
        <div className="w-full space-y-2">
          <div className="flex justify-between items-center text-[11px] font-bold text-gray-500 uppercase tracking-wider">
            <span>System Initialization</span>
            <span className="text-factory-400">{progress}%</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-white/10 overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-factory-500 via-purple-500 to-cyan-400 rounded-full"
              initial={{ width: '0%' }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.4 }}
            />
          </div>
        </div>

        {/* Step List Indicators */}
        <div className="grid grid-cols-6 gap-2 w-full pt-2">
          {STEPS.map((step, idx) => (
            <div
              key={idx}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                idx <= currentStep ? 'bg-cyan-400 shadow-sm shadow-cyan-400/50' : 'bg-white/10'
              }`}
            />
          ))}
        </div>
      </div>

      {/* Footer Info */}
      <div className="absolute bottom-6 text-[10px] text-gray-600 font-bold uppercase tracking-widest">
        Predictive Maintenance Platform • v2.0.0
      </div>
    </motion.div>
  );
}
