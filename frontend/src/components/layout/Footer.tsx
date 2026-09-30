import { Github, Activity, ShieldCheck } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="mt-12 py-6 border-t border-white/[0.06] text-xs text-gray-500 flex flex-col sm:flex-row items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <span className="font-bold text-gray-400">Smart Factory PDM</span>
        <span className="text-gray-700">•</span>
        <span className="text-[11px] font-semibold text-cyan-400 bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded-full">
          v2.0.0 (Enterprise)
        </span>
        <span className="text-gray-700">•</span>
        <span className="text-[11px] font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full flex items-center gap-1">
          <ShieldCheck className="w-3 h-3" /> PRODUCTION
        </span>
      </div>

      <div className="flex items-center gap-6 text-[11px]">
        <span>Build: {new Date().toISOString().split('T')[0]}</span>
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-gray-400 hover:text-white transition-colors"
        >
          <Github className="w-3.5 h-3.5" />
          <span>GitHub Repository</span>
        </a>
      </div>
    </footer>
  );
}
