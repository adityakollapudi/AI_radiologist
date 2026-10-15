import React, { useState, useMemo, useEffect } from 'react';
import { Eye, EyeOff, Sliders, Columns, Layers, Maximize2, Minimize2, Sparkles } from 'lucide-react';
import { isDiseasePositive } from '../../utils/diseaseStatus';

export default function ScanViewer({
  originalScanUrl,
  gradcamScanUrl,
  modality = 'Scan',
  visualizations = null,
  prediction = '',
  metadata = null,
}) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [opacity, setOpacity] = useState(0.80);
  const [viewMode, setViewMode] = useState('single'); // 'single' | 'side-by-side'
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedViewId, setSelectedViewId] = useState('overlay');

  // Extract available distinct diagnostic views
  const availableViews = useMemo(() => {
    const views = [];

    // 1. Original / Input Slice
    const origUrl = visualizations?.original || originalScanUrl;
    if (origUrl) {
      views.push({
        id: 'original',
        label: 'Original Scan',
        badge: 'RAW',
        subtitle: 'Input Slice',
        url: origUrl,
        isBase: true,
      });
    }

    // 2. Cropped Brain / Parenchyma Contour (MRI tumor pipeline)
    if (visualizations?.cropped) {
      views.push({
        id: 'cropped',
        label: 'Cropped Brain',
        badge: 'CROP',
        subtitle: 'Brain Parenchyma',
        url: visualizations.cropped,
        isBase: false,
      });
    }

    // 3. Pure Activation Heatmap
    if (visualizations?.heatmap) {
      views.push({
        id: 'heatmap',
        label: 'Activation Map',
        badge: 'HEAT',
        subtitle: 'Pure Heatmap',
        url: visualizations.heatmap,
        isBase: false,
      });
    }

    // 4. Co-Registered Grad-CAM Overlay
    const overlayUrl = visualizations?.overlay || gradcamScanUrl;
    if (overlayUrl && overlayUrl !== origUrl) {
      views.push({
        id: 'overlay',
        label: 'Grad-CAM Overlay',
        badge: 'OVERLAY',
        subtitle: 'Co-Registered',
        url: overlayUrl,
        isOverlay: true,
      });
    }

    // 5. Multi-Layer Fused Overlay (Dementia MBConv multi-layer fusion)
    if (visualizations?.fused_overlay && visualizations.fused_overlay !== overlayUrl) {
      views.push({
        id: 'fused',
        label: 'Fused Overlay',
        badge: 'FUSION',
        subtitle: 'Multi-Layer CAM',
        url: visualizations.fused_overlay,
        isOverlay: true,
      });
    }

    return views;
  }, [visualizations, originalScanUrl, gradcamScanUrl]);

  // Set default selection to overlay if available, else first view
  useEffect(() => {
    if (availableViews.length > 0) {
      const hasOverlay = availableViews.some((v) => v.id === 'overlay');
      setSelectedViewId(hasOverlay ? 'overlay' : availableViews[0].id);
    }
  }, [availableViews]);

  const activeView = useMemo(() => {
    return availableViews.find((v) => v.id === selectedViewId) || availableViews[0] || null;
  }, [availableViews, selectedViewId]);

  const rawOriginalUrl = visualizations?.original || originalScanUrl;
  const rawOverlayUrl = visualizations?.overlay || gradcamScanUrl;
  const hasOverlayCapability = Boolean(rawOverlayUrl && rawOverlayUrl !== rawOriginalUrl);

  return (
    <div
      className={`glass-panel rounded-2xl p-4 sm:p-6 border border-cyan-500/20 text-left space-y-4 transition-all ${
        isFullscreen ? 'fixed inset-3 sm:inset-6 z-50 overflow-y-auto bg-black/95 shadow-2xl' : ''
      }`}
    >
      {/* 1. Header (Strictly matching reference: Left: GRAD-CAM ACTIVATION HEATMAP, Right: Salient Weights) */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-cyan-500/20">
        <div className="flex items-center gap-2.5">
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(6,182,212,0.8)] animate-pulse" />
          <h3 className="text-xs sm:text-sm font-bold text-cyan-400 tracking-wider uppercase font-heading">
            GRAD-CAM ACTIVATION HEATMAP
          </h3>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-[11px] sm:text-xs font-mono font-medium text-cyan-400/90 tracking-wider">
            Salient Weights
          </span>

          <button
            type="button"
            onClick={() => setIsFullscreen(!isFullscreen)}
            aria-label="Toggle Fullscreen"
            className="p-1.5 rounded-lg text-slate-400 hover:text-cyan-300 hover:bg-cyan-500/10 border border-white/10 transition-all"
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Viewer'}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* 2. Viewer Controls Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-black/50 px-3.5 py-2.5 rounded-xl border border-white/5 text-xs">
        {/* Active View Label & Info */}
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 font-mono text-[10px] uppercase font-semibold">
            {activeView?.badge || 'VIEW'}
          </span>
          <span className="text-slate-300 font-mono text-xs font-medium">
            {activeView?.label || 'Primary View'}
          </span>
          {prediction && (
            <span className="text-slate-500 font-mono text-[11px] hidden sm:inline">
              • Target:{' '}
              <span className={isDiseasePositive(prediction) ? 'text-red-400 font-semibold' : 'text-emerald-400 font-semibold'}>
                {prediction}
              </span>
            </span>
          )}
        </div>

        {/* View Mode & Toggles */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {/* Mode Switcher: Single Dominant vs Side-by-Side */}
          <div className="flex items-center rounded-lg bg-black/80 p-0.5 border border-white/10 text-xs">
            <button
              type="button"
              onClick={() => setViewMode('single')}
              className={`px-2.5 py-1 rounded flex items-center gap-1.5 transition-all ${
                viewMode === 'single'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 font-medium'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Primary</span>
            </button>
            <button
              type="button"
              onClick={() => setViewMode('side-by-side')}
              className={`px-2.5 py-1 rounded flex items-center gap-1.5 transition-all ${
                viewMode === 'side-by-side'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 font-medium'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Columns className="w-3.5 h-3.5" />
              <span>Side-by-Side</span>
            </button>
          </div>

          {/* Heatmap Toggle */}
          {hasOverlayCapability && (
            <button
              type="button"
              onClick={() => setShowHeatmap(!showHeatmap)}
              className={`px-3 py-1 rounded-lg text-xs font-mono font-medium border flex items-center gap-1.5 transition-all ${
                showHeatmap
                  ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 shadow-[0_0_10px_rgba(6,182,212,0.2)]'
                  : 'bg-white/5 text-slate-400 border-white/10 hover:text-white'
              }`}
            >
              {showHeatmap ? <Eye className="w-3.5 h-3.5 text-cyan-400" /> : <EyeOff className="w-3.5 h-3.5" />}
              <span>Heatmap {showHeatmap ? 'ON' : 'OFF'}</span>
            </button>
          )}
        </div>
      </div>

      {/* 3. Opacity Slider (When overlay view or heatmap active) */}
      {hasOverlayCapability && showHeatmap && (activeView?.isOverlay || viewMode === 'side-by-side') && (
        <div className="flex items-center gap-3 bg-black/40 px-4 py-2 rounded-xl border border-white/5 text-xs font-mono">
          <Sliders className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <span className="text-slate-400">Grad-CAM Opacity:</span>
          <input
            type="range"
            min="0.10"
            max="1.00"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-32 sm:w-48 accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
          <span className="text-cyan-400 font-bold w-12 text-right">
            {(opacity * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* 4. Primary Image Viewer Stage */}
      {viewMode === 'single' ? (
        /* Dominant Focused Medical Viewer */
        <div className="w-full max-w-xl mx-auto rounded-2xl bg-black/90 border border-cyan-500/30 p-2 sm:p-3 relative shadow-[0_0_30px_rgba(6,182,212,0.08)]">
          {/* Main Display Box with Aspect-Ratio Containment */}
          <div className="w-full aspect-square max-h-[500px] rounded-xl overflow-hidden bg-slate-950 relative flex items-center justify-center border border-white/10">
            {activeView?.isOverlay ? (
              /* Aligned Layered Visualization */
              <div className="relative w-full h-full flex items-center justify-center overflow-hidden">
                {rawOriginalUrl && (
                  <img
                    src={rawOriginalUrl}
                    alt="Original Base Scan"
                    className="w-full h-full object-contain filter contrast-125 select-none"
                  />
                )}
                {activeView.url && (
                  <img
                    src={activeView.url}
                    alt="Grad-CAM Overlay"
                    style={{ opacity: showHeatmap ? opacity : 0 }}
                    className="w-full h-full object-contain absolute inset-0 select-none transition-opacity duration-200 pointer-events-none"
                  />
                )}
              </div>
            ) : activeView?.url ? (
              /* Single View Mode (e.g. Original Scan, Cropped Brain, Pure Heatmap) */
              <div className="relative w-full h-full flex items-center justify-center overflow-hidden">
                <img
                  src={activeView.url}
                  alt={activeView.label}
                  className="w-full h-full object-contain filter contrast-125 select-none"
                />
                {/* Optional overlay onto base if user turns on heatmap on base view */}
                {activeView.isBase && hasOverlayCapability && showHeatmap && (
                  <img
                    src={rawOverlayUrl}
                    alt="Grad-CAM Overlay"
                    style={{ opacity }}
                    className="w-full h-full object-contain absolute inset-0 select-none transition-opacity duration-200 pointer-events-none"
                  />
                )}
              </div>
            ) : (
              <div className="text-xs text-slate-500 font-mono text-center p-6">
                No visualization image available
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Side-by-Side Comparison Layout */
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* Panel 1: Original Input Scan */}
          <div className="rounded-xl bg-black/80 border border-white/10 p-3 relative group">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono uppercase tracking-wider text-slate-300 font-semibold">
                Original Input Scan
              </span>
              <span className="text-[10px] font-mono text-slate-500">Unprocessed</span>
            </div>
            <div className="w-full aspect-square rounded-lg overflow-hidden bg-slate-950 flex items-center justify-center border border-white/5">
              {rawOriginalUrl ? (
                <img
                  src={rawOriginalUrl}
                  alt="Original Medical Radiograph"
                  className="w-full h-full object-contain filter contrast-125 select-none"
                />
              ) : (
                <span className="text-xs text-slate-600">No original scan</span>
              )}
            </div>
          </div>

          {/* Panel 2: Selected Diagnostic View / Heatmap */}
          <div className="rounded-xl bg-black/80 border border-cyan-500/25 p-3 relative group">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono uppercase tracking-wider text-cyan-400 font-semibold">
                {activeView?.label || 'Grad-CAM Feature Map'}
              </span>
              <span className="text-[10px] font-mono text-cyan-400/80">Salient Weights</span>
            </div>
            <div className="w-full aspect-square rounded-lg overflow-hidden bg-slate-950 relative flex items-center justify-center border border-cyan-500/10">
              {rawOriginalUrl && (
                <img
                  src={rawOriginalUrl}
                  alt="Background Radiograph"
                  className="w-full h-full object-contain filter contrast-125 select-none"
                />
              )}
              {rawOverlayUrl && showHeatmap && (
                <img
                  src={activeView?.isOverlay ? activeView.url : rawOverlayUrl}
                  alt="Grad-CAM Activation Map"
                  style={{ opacity }}
                  className="w-full h-full object-contain absolute inset-0 select-none transition-opacity duration-200 pointer-events-none"
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* 5. Dedicated Horizontal Thumbnail Strip (Below Primary Viewer) */}
      {availableViews.length > 1 && (
        <div className="pt-2 space-y-2 border-t border-white/10">
          <div className="flex items-center justify-between px-1">
            <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Sparkles className="w-3 h-3 text-cyan-400" />
              <span>Related Diagnostic Views</span>
            </span>
            <span className="text-[10px] font-mono text-cyan-400/80">
              {availableViews.length} available views • Click thumbnail to inspect
            </span>
          </div>

          {/* Horizontal scrollable strip container */}
          <div className="flex items-center gap-3 sm:gap-4 overflow-x-auto pb-2 pt-1 px-1 scrollbar-thin">
            {availableViews.map((view) => {
              const isSelected = view.id === selectedViewId;
              return (
                <button
                  key={view.id}
                  type="button"
                  onClick={() => setSelectedViewId(view.id)}
                  className={`group relative flex flex-col items-center gap-1.5 shrink-0 focus:outline-none transition-all ${
                    isSelected ? 'scale-[1.02]' : 'hover:scale-[1.01]'
                  }`}
                >
                  {/* Thumbnail Card */}
                  <div
                    className={`w-20 h-20 sm:w-24 sm:h-24 aspect-square rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center relative transition-all ${
                      isSelected
                        ? 'border-2 border-cyan-400 ring-2 ring-cyan-400/30 shadow-[0_0_15px_rgba(6,182,212,0.45)]'
                        : 'border border-white/10 opacity-70 hover:opacity-100 hover:border-cyan-500/40'
                    }`}
                  >
                    <img
                      src={view.url}
                      alt={view.label}
                      className="w-full h-full object-contain filter contrast-125 select-none"
                    />

                    {/* Badge */}
                    <span className="absolute bottom-1 right-1 px-1 py-0.2 rounded bg-black/80 text-[8px] font-mono text-cyan-300 border border-cyan-500/30 font-bold">
                      {view.badge}
                    </span>

                    {/* Active Selected Indicator */}
                    {isSelected && (
                      <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_6px_rgba(6,182,212,1)] animate-pulse" />
                    )}
                  </div>

                  {/* Caption & Subtitle */}
                  <div className="text-center w-20 sm:w-24">
                    <span
                      className={`block text-[11px] font-mono font-medium truncate transition-colors ${
                        isSelected ? 'text-cyan-300 font-bold' : 'text-slate-400 group-hover:text-slate-200'
                      }`}
                    >
                      {view.label}
                    </span>
                    <span className="block text-[9px] font-mono text-slate-500 truncate">
                      {view.subtitle}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* 6. Heatmap Colorbar Legend */}
      {hasOverlayCapability && showHeatmap && (
        <div className="flex items-center justify-between pt-2 px-1 text-[10px] font-mono text-slate-400 border-t border-white/5">
          <span>Low Activation (0.0)</span>
          <div className="w-36 sm:w-56 h-2 rounded bg-gradient-to-r from-blue-900 via-cyan-400 via-yellow-400 to-red-600 border border-white/10" />
          <span>High Salience (1.0)</span>
        </div>
      )}
    </div>
  );
}
