import { motion } from 'framer-motion';
import {
  Factory,
  Brain,
  Cpu,
  Database,
  Layers,
  ShieldCheck,
  GitBranch,
  Terminal,
  Code,
  Globe,
  Award,
  BookOpen,
  CheckCircle2,
} from 'lucide-react';
import ErrorBoundary from '../components/ui/ErrorBoundary';

const TECH_STACK = [
  { category: 'Machine Learning', items: ['Scikit-Learn', 'XGBoost', 'SHAP', 'NumPy', 'Pandas', 'Joblib'] },
  { category: 'MLOps & Tracking', items: ['MLflow v2.10', 'SQLite Registry', 'Evidently AI', 'Artifact Manifest'] },
  { category: 'Backend & Infrastructure', items: ['Python 3.10', 'Flask REST API', 'Gunicorn WSGI', 'Docker & Compose', 'Nginx'] },
  { category: 'Frontend & UI/UX', items: ['React 18', 'TypeScript', 'Vite 5', 'Tailwind CSS', 'Framer Motion', 'Recharts'] },
];

const MODEL_MATRIX = [
  { name: 'XGBoost Classifier', accuracy: '98.4%', precision: '97.6%', recall: '96.8%', f1: '97.2%', status: 'Active Champion' },
  { name: 'Random Forest', accuracy: '97.8%', precision: '96.2%', recall: '95.9%', f1: '96.0%', status: 'Challenger' },
  { name: 'Decision Tree', accuracy: '94.2%', precision: '92.1%', recall: '91.8%', f1: '91.9%', status: 'Evaluated' },
  { name: 'Logistic Regression', accuracy: '89.6%', precision: '86.4%', recall: '85.2%', f1: '85.8%', status: 'Baseline' },
];

function AboutPageContent() {
  return (
    <div className="space-y-8">
      {/* Hero Banner */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-8 bg-gradient-to-r from-factory-500/10 via-purple-600/10 to-cyan-500/10 border border-white/10 relative overflow-hidden"
      >
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-extrabold px-3 py-1 rounded-full bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 uppercase tracking-widest flex items-center gap-1.5">
              <Award className="w-3.5 h-3.5" /> Industry 4.0 Portfolio Project
            </span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            Smart Factory Predictive Maintenance Platform
          </h1>

          <p className="text-sm text-gray-300 leading-relaxed font-normal">
            An end-to-end, enterprise-grade MLOps system engineered to forecast equipment failures before downtime strikes. Powered by continuous machine telemetry analysis, SHAP explainability, and automated drift detection.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-gray-300">
              <ShieldCheck className="w-4 h-4 text-emerald-400" /> Version 2.0.0 Production Certified
            </div>
            <div className="flex items-center gap-2 text-xs font-semibold text-gray-300">
              <Globe className="w-4 h-4 text-cyan-400" /> Azure PdM Dataset (100 Machines)
            </div>
          </div>
        </div>
      </motion.div>

      {/* Technology Stack Grid */}
      <div className="space-y-4">
        <h2 className="text-sm font-bold text-white tracking-tight uppercase tracking-wider text-gray-400 flex items-center gap-2">
          <Code className="w-4 h-4 text-factory-400" /> Enterprise Technology Stack
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {TECH_STACK.map((group, idx) => (
            <motion.div
              key={group.category}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: idx * 0.1 }}
              className="glass-card p-5 space-y-3"
            >
              <h3 className="text-xs font-bold text-cyan-400 uppercase tracking-wider">{group.category}</h3>
              <div className="flex flex-wrap gap-2">
                {group.items.map((item) => (
                  <span
                    key={item}
                    className="text-[11px] font-semibold px-2.5 py-1 rounded-lg bg-white/[0.04] border border-white/[0.08] text-gray-300"
                  >
                    {item}
                  </span>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Architecture Flow Diagram */}
      <div className="glass-card p-6 space-y-5">
        <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <Layers className="w-4 h-4 text-purple-400" /> End-to-End System Architecture
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 text-center">
          {[
            { step: '1. Ingestion', desc: 'Azure PdM Telemetry & Sensor Streams', icon: Database, color: 'text-indigo-400' },
            { step: '2. Pipeline', desc: 'Rolling Features & Scaling (Joblib)', icon: Cpu, color: 'text-cyan-400' },
            { step: '3. Inference', desc: 'Flask + Gunicorn WSGI Server', icon: Terminal, color: 'text-emerald-400' },
            { step: '4. Tracking', desc: 'MLflow Registry & SQLite DB', icon: GitBranch, color: 'text-purple-400' },
            { step: '5. Dashboard', desc: 'React 18 + Vite Glassmorphism UI', icon: Factory, color: 'text-factory-400' },
          ].map((s, idx) => (
            <div key={s.step} className="glass-card p-4 bg-white/[0.02] space-y-2 border border-white/[0.06]">
              <s.icon className={`w-6 h-6 ${s.color} mx-auto`} />
              <h4 className="text-xs font-bold text-white">{s.step}</h4>
              <p className="text-[10px] text-gray-400 leading-snug">{s.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Model Performance Matrix */}
      <div className="glass-card p-6 space-y-4">
        <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <Brain className="w-4 h-4 text-cyan-400" /> Benchmark Model Performance Matrix
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-white/[0.08] text-gray-500 text-[10px] font-bold uppercase tracking-wider">
                <th className="pb-3">Model Architecture</th>
                <th className="pb-3">Accuracy</th>
                <th className="pb-3">Precision</th>
                <th className="pb-3">Recall</th>
                <th className="pb-3">F1 Score</th>
                <th className="pb-3 text-right">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] text-xs text-gray-300">
              {MODEL_MATRIX.map((m) => (
                <tr key={m.name} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-3.5 font-bold text-white flex items-center gap-2">
                    {m.name}
                  </td>
                  <td className="py-3.5 font-semibold">{m.accuracy}</td>
                  <td className="py-3.5">{m.precision}</td>
                  <td className="py-3.5">{m.recall}</td>
                  <td className="py-3.5 font-extrabold text-cyan-400">{m.f1}</td>
                  <td className="py-3.5 text-right">
                    <span
                      className={`text-[10px] font-bold px-2.5 py-0.5 rounded-full border uppercase tracking-wider ${
                        m.status === 'Active Champion'
                          ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                          : 'bg-white/[0.04] text-gray-400 border-white/[0.08]'
                      }`}
                    >
                      {m.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function AboutPage() {
  return (
    <ErrorBoundary>
      <AboutPageContent />
    </ErrorBoundary>
  );
}
