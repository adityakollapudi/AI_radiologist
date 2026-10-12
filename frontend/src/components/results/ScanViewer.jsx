import React, { useState } from 'react';
import { Eye, EyeOff, Sliders, Columns, Layers, Maximize2 } from 'lucide-react';

export default function ScanViewer({
  originalScanUrl,
  gradcamScanUrl,
  _modality = 'Scan',
}) {

  const [showHeatmap, setShowHeatmap] = useState(true);
  const [opacity, setOpacity] = useState(0.75);
  const [viewMode, setViewMode] = useState('side-by-side'); // 'side-by-side' | 'overlay'
  const [isFullscreen, setIsFullscreen] = useState(false);

  const hasGradcam = Boolean(gradcamScanUrl);

  return (
    <div
      className={`glass-panel rounded-2xl p-5 sm:p-6 border border-cyan-500/20 text-left space-y-4 ${
        isFullscreen ? 'fixed inset-4 z-50 overflow-y-auto bg-black/95' : ''
      }`}
    >
      {/* Top Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-white/10">
        <div>
          <h3 className="text-sm sm:text-base font-semibold text-white tracking-wide font-heading flex items-center gap-2">
            <span>Radiology Scan & Grad-CAM Attention</span>
          </h3>
          <p className="text-[11px] text-slate-400 font-mono">
            Convolutional layer gradient attribution mapping
          </p>
        </div>

        {/* View mode switches & sliders */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {/* Mode Toggle: Side-by-Side vs Overlay */}
          <div className="flex items-center rounded-lg bg-black/60 p-1 border border-white/10 text-xs">
            <button
              type="button"
              onClick={() => setViewMode('side-by-side')}
              className={`px-2.5 py-1 rounded flex items-center gap-1.5 transition-all ${
                viewMode === 'side-by-side'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Columns className="w-3.5 h-3.5" />
              <span>Side-by-Side</span>
            </button>
            <button
              type="button"
              onClick={() => setViewMode('overlay')}
              className={`px-2.5 py-1 rounded flex items-center gap-1.5 transition-all ${
                viewMode === 'overlay'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Overlay</span>
            </button>
          </div>

          {/* Heatmap Toggle */}
          {hasGradcam && (
            <button
              type="button"
              onClick={() => setShowHeatmap(!showHeatmap)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition-all ${
                showHeatmap
                  ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30'
                  : 'bg-white/5 text-slate-400 border-white/10 hover:text-white'
              }`}
            >
              {showHeatmap ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
              <span>Heatmap {showHeatmap ? 'ON' : 'OFF'}</span>
            </button>
          )}

          {/* Fullscreen Toggle */}
          <button
            type="button"
            onClick={() => setIsFullscreen(!isFullscreen)}
            aria-label="Toggle Fullscreen"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white bg-white/5 border border-white/10"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Opacity Slider (When overlay active or side-by-side with heatmap) */}
      {hasGradcam && showHeatmap && (
        <div className="flex items-center gap-3 bg-black/40 px-4 py-2 rounded-xl border border-white/5 text-xs font-mono">
          <Sliders className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <span className="text-slate-400">Grad-CAM Opacity:</span>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-32 sm:w-48 accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
          <span className="text-cyan-400 font-bold w-10 text-right">
            {(opacity * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* Main Image Viewer Stage */}
      {viewMode === 'side-by-side' ? (
        /* Side by Side Layout */
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Panel 1: Original Scan */}
          <div className="rounded-xl bg-black/80 border border-white/10 p-3 relative group">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono uppercase tracking-wider text-slate-300">
                Original Input Scan
              </span>
              <span className="text-[10px] font-mono text-slate-500">Unprocessed</span>
            </div>
            <div className="w-full aspect-square rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center border border-white/5">
              {originalScanUrl ? (
                <img
                  src={originalScanUrl}
                  alt="Original Medical Radiograph"
                  className="w-full h-full object-contain filter contrast-125"
                />
              ) : (
                <span className="text-xs text-slate-600">No original scan</span>
              )}
            </div>
          </div>

          {/* Panel 2: Grad-CAM Feature Map */}
          <div className="rounded-xl bg-black/80 border border-cyan-500/25 p-3 relative group">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono uppercase tracking-wider text-cyan-400 font-semibold">
                Grad-CAM Activation Heatmap
              </span>
              <span className="text-[10px] font-mono text-cyan-400/80">Salient Weights</span>
            </div>
            <div className="w-full aspect-square rounded-lg overflow-hidden bg-slate-950 relative flex items-center justify-center border border-cyan-500/10">
              {/* Base scan underneath */}
              {originalScanUrl && (
                <img
                  src={originalScanUrl}
                  alt="Background Radiograph"
                  className="w-full h-full object-contain filter contrast-125 absolute inset-0"
                />
              )}
              {/* Heatmap overlay */}
              {hasGradcam && showHeatmap ? (
                <img
                  src={gradcamScanUrl}
                  alt="Grad-CAM Activation Map"
                  style={{ opacity }}
                  className="w-full h-full object-contain absolute inset-0 mix-blend-screen transition-opacity duration-200"
                />
              ) : (
                !hasGradcam && (
                  <div className="text-xs text-slate-500 font-mono text-center p-4">
                    Grad-CAM unavailable for this output
                  </div>
                )
              )}
            </div>
          </div>
        </div>
      ) : (
        /* Overlay Layout */
        <div className="w-full max-w-lg mx-auto rounded-xl bg-black/90 border border-cyan-500/30 p-3 relative">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-mono uppercase tracking-wider text-cyan-400 font-semibold">
              Co-Registered Radiographic Overlay
            </span>
            <span className="text-[10px] font-mono text-slate-400">
              Opacity: {(opacity * 100).toFixed(0)}%
            </span>
          </div>

          <div className="w-full aspect-square rounded-lg overflow-hidden bg-slate-950 relative flex items-center justify-center border border-white/10 shadow-2xl">
            {originalScanUrl && (
              <img
                src={originalScanUrl}
                alt="Base Scan"
                className="w-full h-full object-contain filter contrast-125"
              />
            )}
            {hasGradcam && showHeatmap && (
              <img
                src={gradcamScanUrl}
                alt="Grad-CAM Overlay"
                style={{ opacity }}
                className="w-full h-full object-contain absolute inset-0 mix-blend-screen transition-opacity duration-200"
              />
            )}
          </div>
        </div>
      )}

      {/* Heatmap Colorbar Legend */}
      {hasGradcam && showHeatmap && (
        <div className="flex items-center justify-between pt-2 px-1 text-[10px] font-mono text-slate-400">
          <span>Low Activation (0.0)</span>
          <div className="w-36 sm:w-48 h-2 rounded bg-gradient-to-r from-blue-900 via-cyan-400 via-yellow-400 to-red-600 border border-white/10" />
          <span>High Salience (1.0)</span>
        </div>
      )}
    </div>
  );
}
