import React from 'react';
import { Activity, Binary } from 'lucide-react';


const PHASES = [
  { key: 'preparing', text: 'PREPARING IMAGE', sub: 'Normalizing tensor dimensions and contrast normalization' },
  { key: 'running', text: 'RUNNING AI MODEL', sub: 'Forward pass through deep convolutional neural network' },
  { key: 'gradcam', text: 'GENERATING GRAD-CAM', sub: 'Computing gradients at final convolutional layer' },
  { key: 'complete', text: 'ANALYSIS COMPLETE', sub: 'Aggregating class probabilities and feature heatmaps' },
];

export default function AnalysisLoader({ phase = 'preparing', previewUrl }) {
  const currentPhaseIndex = Math.max(
    0,
    PHASES.findIndex((p) => p.key === phase)
  );

  return (
    <div className="w-full max-w-xl mx-auto py-10 px-4 text-center space-y-8 animate-fadeIn">
      {/* 3D Medical Scan Animation Frame */}
      <div className="relative w-64 h-64 sm:w-72 sm:h-72 mx-auto rounded-2xl bg-black/90 border-2 border-cyan-500/40 p-2 shadow-[0_0_50px_rgba(6,182,212,0.25)] overflow-hidden">
        {/* Corner HUD reticles */}
        <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-cyan-400 z-20 pointer-events-none" />
        <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-cyan-400 z-20 pointer-events-none" />
        <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-cyan-400 z-20 pointer-events-none" />
        <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-cyan-400 z-20 pointer-events-none" />

        {/* Scan image being analyzed */}
        {previewUrl ? (
          <img
            src={previewUrl}
            alt="Active scan analysis"
            className="w-full h-full object-contain filter contrast-125 opacity-80"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-cyan-400/40">
            <Binary className="w-16 h-16 animate-pulse" />
          </div>
        )}

        {/* Sweeping Laser Scan Line */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="w-full h-1.5 bg-gradient-to-r from-transparent via-cyan-300 to-transparent shadow-[0_0_20px_#22d3ee] animate-scan-sweep" />
        </div>

        {/* Holographic grid overlay */}
        <div className="absolute inset-0 scan-line-overlay opacity-40 pointer-events-none" />

        {/* Central HUD Target Indicator */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-28 h-28 rounded-full border border-cyan-400/20 border-dashed animate-spin" style={{ animationDuration: '10s' }} />
        </div>
      </div>

      {/* Dynamic Text Phases */}
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-300 text-xs font-mono uppercase tracking-widest">
          <Activity className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
          <span>Inference Engine Active</span>
        </div>

        <h2 className="text-xl sm:text-2xl font-bold text-white tracking-wider font-heading uppercase text-cyan-300">
          {PHASES[currentPhaseIndex]?.text || 'PROCESSING SCAN'}
        </h2>

        <p className="text-xs sm:text-sm text-slate-400 font-mono max-w-md mx-auto">
          {PHASES[currentPhaseIndex]?.sub}
        </p>
      </div>

      {/* Step Indicators (No fake numerical percent) */}
      <div className="grid grid-cols-4 gap-2 max-w-md mx-auto pt-2">
        {PHASES.map((p, idx) => {
          const isDone = idx < currentPhaseIndex;
          const isCurrent = idx === currentPhaseIndex;
          return (
            <div key={p.key} className="space-y-1.5">
              <div
                className={`h-1.5 rounded-full transition-all duration-500 ${
                  isDone
                    ? 'bg-cyan-400 shadow-[0_0_8px_#22d3ee]'
                    : isCurrent
                    ? 'bg-cyan-400/80 animate-pulse shadow-[0_0_12px_#06b6d4]'
                    : 'bg-white/10'
                }`}
              />
              <span
                className={`text-[10px] font-mono block truncate ${
                  isCurrent ? 'text-cyan-300 font-semibold' : isDone ? 'text-slate-400' : 'text-slate-600'
                }`}
              >
                0{idx + 1}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
