import React, { useEffect, useState } from 'react';
import { BarChart3 } from 'lucide-react';

export default function ProbabilityBars({ probabilities = {}, predictedClass }) {
  const [animatedWidths, setAnimatedWidths] = useState({});

  useEffect(() => {
    // Trigger smooth bar growth animation on mount or update
    const timer = setTimeout(() => {
      setAnimatedWidths(probabilities);
    }, 120);
    return () => clearTimeout(timer);
  }, [probabilities]);

  const entries = Object.entries(probabilities);
  if (entries.length === 0) return null;

  return (
    <div className="glass-panel rounded-2xl p-6 border border-white/10 text-left space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-white tracking-wide flex items-center gap-2 font-mono uppercase">
          <BarChart3 className="w-4 h-4 text-cyan-400" />
          <span>Class Probabilities</span>
        </h3>
        <span className="text-[11px] font-mono text-slate-500">Softmax Distribution</span>
      </div>

      <div className="space-y-3.5">
        {entries.map(([className, rawVal]) => {
          const val = typeof rawVal === 'number' ? rawVal : parseFloat(rawVal) || 0;
          const pct = (val * 100).toFixed(1);
          const isMax = className === predictedClass;
          const currentWidth = animatedWidths[className] !== undefined
            ? `${(animatedWidths[className] * 100).toFixed(1)}%`
            : '0%';

          return (
            <div key={className} className="space-y-1.5">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className={`font-medium ${isMax ? 'text-cyan-300 font-bold' : 'text-slate-300'}`}>
                  {className}
                </span>
                <span className={`font-mono ${isMax ? 'text-cyan-400 font-semibold' : 'text-slate-400'}`}>
                  {pct}%
                </span>
              </div>

              {/* Bar track */}
              <div className="w-full h-2 rounded-full bg-black/60 border border-white/5 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-1000 ease-out ${
                    isMax
                      ? 'bg-gradient-to-r from-cyan-400 to-blue-500 shadow-[0_0_12px_rgba(6,182,212,0.6)]'
                      : 'bg-slate-700/80'
                  }`}
                  style={{ width: currentWidth }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
