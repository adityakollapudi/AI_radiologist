import React from 'react';
import { Play, RefreshCw, CheckCircle2 } from 'lucide-react';


export default function FilePreview({
  previewUrl,
  fileName = 'Medical Scan',
  fileSize,
  modality,
  onAnalyze,
  onChangeScan,
  isAnalyzing,
}) {
  return (
    <div className="w-full space-y-6">
      <div className="glass-panel rounded-2xl p-4 sm:p-6 border border-cyan-500/25 relative overflow-hidden">
        {/* Subtle scan grid backdrop */}
        <div className="absolute inset-0 scan-line-overlay opacity-30 pointer-events-none" />

        <div className="flex flex-col sm:flex-row items-center gap-6 relative z-10">
          {/* Image Display */}
          <div className="w-48 h-48 sm:w-56 sm:h-56 rounded-xl bg-black/80 border border-cyan-500/30 overflow-hidden flex items-center justify-center relative shadow-lg group">
            <img
              src={previewUrl}
              alt="Medical Scan Preview"
              className="w-full h-full object-contain filter contrast-125"
            />
            <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur-sm border border-white/10 text-[10px] font-mono text-cyan-300">
              ORIGINAL INPUT
            </div>
          </div>

          {/* Details & Actions */}
          <div className="flex-1 text-center sm:text-left space-y-3">
            <div className="flex items-center justify-center sm:justify-start gap-2">
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-mono">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Scan Loaded
              </span>
              <span className="text-xs uppercase font-mono text-slate-400 bg-white/5 px-2 py-0.5 rounded border border-white/5">
                {modality}
              </span>
            </div>

            <h3 className="text-lg font-bold text-white font-heading truncate max-w-sm">
              {fileName}
            </h3>

            {fileSize && (
              <p className="text-xs text-slate-400 font-mono">
                Size: {(fileSize / 1024).toFixed(1)} KB
              </p>
            )}

            <p className="text-xs text-slate-400 leading-relaxed max-w-md">
              Ready for deep learning inference. The image will be preprocessed to model dimensions, passed through the classification backbone, and Grad-CAM attention will be synthesized.
            </p>

            <div className="pt-2 flex flex-wrap items-center justify-center sm:justify-start gap-3">
              <button
                type="button"
                onClick={onAnalyze}
                disabled={isAnalyzing}
                className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs uppercase tracking-wider shadow-[0_0_20px_rgba(6,182,212,0.4)] transition-all flex items-center gap-2 group cursor-pointer"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Analyze Scan</span>
              </button>

              <button
                type="button"
                onClick={onChangeScan}
                disabled={isAnalyzing}
                className="px-4 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-slate-300 hover:text-white text-xs font-medium transition-all flex items-center gap-1.5 cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Change Scan</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
