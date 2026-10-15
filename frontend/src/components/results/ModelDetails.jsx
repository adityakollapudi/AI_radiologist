import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Cpu, Database, Award } from 'lucide-react';


export default function ModelDetails({ result }) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!result) return null;

  const {
    model_name,
    framework = 'PyTorch 2.2',
    input_size = '224x224 px',
    threshold = 0.5,
    dataset = 'Standard Medical Imaging Benchmark',
    metrics = {},
  } = result;

  return (
    <div className="glass-panel rounded-2xl border border-white/10 text-left overflow-hidden transition-all">
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-4 sm:p-5 flex items-center justify-between text-left hover:bg-white/5 transition-colors focus:outline-none"
      >
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white font-heading tracking-wide">
              MODEL DETAILS
            </h3>
            <p className="text-[11px] text-slate-400 font-mono">
              Verified neural network specifications and validation benchmarks
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-cyan-400">
          <span>{isExpanded ? 'Hide Specs' : 'View Specs'}</span>
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {isExpanded && (
        <div className="p-5 pt-0 border-t border-white/5 space-y-4 animate-fadeIn">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono pt-3">
            <div className="p-3 rounded-xl bg-slate-950/70 border border-white/5">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Model Name</span>
              <span className="text-white font-semibold">{model_name}</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/70 border border-white/5">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Framework</span>
              <span className="text-white font-semibold">{framework}</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/70 border border-white/5">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Input Tensor</span>
              <span className="text-cyan-300 font-semibold">{input_size}</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/70 border border-white/5">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Decision Threshold</span>
              <span className="text-cyan-300 font-semibold">{(threshold * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Dataset */}
          <div className="p-3 rounded-xl bg-slate-950/70 border border-white/5 text-xs">
            <div className="flex items-center gap-2 text-slate-400 font-mono text-[11px] mb-1">
              <Database className="w-3.5 h-3.5 text-cyan-400" />
              <span>Training & Benchmark Dataset</span>
            </div>
            <p className="text-slate-200 font-mono text-xs">{dataset}</p>
          </div>

          {/* Reported Validation Metrics */}
          {metrics && Object.keys(metrics).length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-slate-400 font-mono text-[11px]">
                <Award className="w-3.5 h-3.5 text-amber-400" />
                <span>Reported Evaluation Metrics</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {Object.entries(metrics).map(([metricKey, metricVal]) => (
                  <div key={metricKey} className="p-2.5 rounded-lg bg-black/40 border border-white/5 font-mono text-center">
                    <span className="text-[10px] text-slate-400 block">{metricKey}</span>
                    <span className="text-sm font-bold text-cyan-300">{metricVal}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
