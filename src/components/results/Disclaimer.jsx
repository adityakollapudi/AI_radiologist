import React from 'react';
import { AlertCircle } from 'lucide-react';

export default function Disclaimer() {
  return (
    <div
      role="note"
      aria-label="Medical Research Disclaimer"
      className="w-full p-4 rounded-xl bg-slate-950/80 border border-amber-500/25 text-left flex items-start gap-3 shadow-lg"
    >
      <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
      <div className="space-y-1">
        <span className="text-[11px] font-mono uppercase font-bold tracking-wider text-amber-400 block">
          Mandatory Medical Disclaimer
        </span>
        <p className="text-xs text-slate-300 leading-relaxed">
          Research and educational use only. This system does not provide a clinical diagnosis and must not replace evaluation by a qualified healthcare professional.
        </p>
      </div>
    </div>
  );
}
