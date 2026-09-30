import { motion } from 'framer-motion';
import { useQuery } from '../hooks/useQuery';
import { fetchModels } from '../services/api';
import { Cpu, Layers, Calendar, Award, CheckCircle } from 'lucide-react';

export default function Models() {
  const { data: modelsData, isLoading } = useQuery(fetchModels);
  const modelInfo = modelsData?.model;

  if (isLoading) {
    return (
      <div className="py-20 text-center space-y-4">
        <div className="w-10 h-10 border-4 border-factory-500 border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-sm text-gray-500">Retrieving model metadata...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-extrabold text-white tracking-tight">Active Production Model</h1>
        <p className="text-xs text-gray-500 mt-1">Review the metadata and status of the current ML model in production.</p>
      </div>

      {modelInfo ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-card p-6 flex flex-col items-start gap-4"
          >
            <div className="p-3 rounded-xl bg-factory-500/10 text-factory-400">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Algorithm</p>
              <p className="text-xl font-bold text-white mt-1">{modelInfo.active_model}</p>
            </div>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-card p-6 flex flex-col items-start gap-4"
          >
            <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400">
              <Award className="w-6 h-6" />
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Model Version</p>
              <p className="text-xl font-bold text-white mt-1">v{modelInfo.version}</p>
            </div>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-card p-6 flex flex-col items-start gap-4"
          >
            <div className="p-3 rounded-xl bg-cyan-500/10 text-cyan-400">
              <Calendar className="w-6 h-6" />
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Training Date</p>
              <p className="text-xl font-bold text-white mt-1">
                {modelInfo.training_date ? new Date(modelInfo.training_date).toLocaleString() : 'N/A'}
              </p>
            </div>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-card p-6 flex flex-col items-start gap-4"
          >
            <div className="p-3 rounded-xl bg-purple-500/10 text-purple-400">
              <Layers className="w-6 h-6" />
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Feature Inputs</p>
              <p className="text-xl font-bold text-white mt-1">{modelInfo.feature_count} features</p>
            </div>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="glass-card p-6 flex flex-col items-start gap-4"
          >
            <div className="p-3 rounded-xl bg-emerald-500/10 text-emerald-400">
              <CheckCircle className="w-6 h-6" />
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Status</p>
              <p className="text-xl font-bold text-white mt-1">{modelInfo.status}</p>
            </div>
          </motion.div>
        </div>
      ) : (
        <div className="py-20 text-center space-y-4 glass-card">
          <p className="text-sm text-gray-400 font-medium">No active production model found.</p>
        </div>
      )}
    </div>
  );
}
