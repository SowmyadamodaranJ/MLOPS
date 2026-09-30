import { motion } from 'framer-motion';
import { useQuery } from '../hooks/useQuery';
import { fetchMetrics } from '../services/api';
import { BarChart3, Target, Activity, Zap, Percent } from 'lucide-react';

export default function Analytics() {
  const { data: metricsData, isLoading } = useQuery(fetchMetrics);
  const metrics = metricsData?.metrics;

  if (isLoading) {
    return (
      <div className="py-20 text-center space-y-4">
        <div className="w-10 h-10 border-4 border-factory-500 border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-sm text-gray-500">Retrieving model metrics...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-extrabold text-white tracking-tight">Model Analytics</h1>
        <p className="text-xs text-gray-500 mt-1">Review the latest evaluation metrics of the active machine learning model.</p>
      </div>

      {metrics ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { label: 'Accuracy', value: metrics.Accuracy, icon: Target, color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
            { label: 'Precision', value: metrics.Precision, icon: Zap, color: 'text-cyan-400', bg: 'bg-cyan-500/10' },
            { label: 'Recall', value: metrics.Recall, icon: Activity, color: 'text-purple-400', bg: 'bg-purple-500/10' },
            { label: 'F1 Score', value: metrics['F1 Score'], icon: BarChart3, color: 'text-factory-400', bg: 'bg-factory-500/10' },
            { label: 'ROC-AUC', value: metrics['ROC-AUC'], icon: Percent, color: 'text-rose-400', bg: 'bg-rose-500/10' },
            { label: 'Balanced Acc', value: metrics['Balanced Accuracy'], icon: Target, color: 'text-blue-400', bg: 'bg-blue-500/10' },
            { label: 'Specificity', value: metrics.Specificity, icon: Activity, color: 'text-amber-400', bg: 'bg-amber-500/10' }
          ].map((metric, index) => (
            <motion.div 
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="glass-card p-6 flex flex-col items-start gap-4"
            >
              <div className={`p-3 rounded-xl ${metric.bg} ${metric.color}`}>
                <metric.icon className="w-6 h-6" />
              </div>
              <div>
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{metric.label}</p>
                <p className="text-2xl font-black text-white mt-1">{(metric.value * 100).toFixed(2)}%</p>
              </div>
            </motion.div>
          ))}
        </div>
      ) : (
        <div className="py-20 text-center space-y-4 glass-card">
          <p className="text-sm text-gray-400 font-medium">Metrics not available.</p>
        </div>
      )}
    </div>
  );
}
