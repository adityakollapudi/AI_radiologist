import React from 'react';
import { AlertTriangle, AlertOctagon, AlertCircle } from 'lucide-react';
import { isDiseasePositive } from '../../utils/diseaseStatus';

export default function ResultCard({ result }) {
  if (!result) return null;

  const {
    prediction,
    confidence,
    modality,
    routed_to,
    threshold = 0.5,
    warnings = [],
  } = result;

  const isUnknown = routed_to === 'unknown' || prediction === 'UNKNOWN';
  const isPositiveFinding = isDiseasePositive(prediction);

  const confidencePct = (confidence * 100).toFixed(1);

  // Modality & Routing Title
  let workflowBadge = 'Standard Pipeline';
  if (modality === 'xray') {
    if (routed_to === 'chest') workflowBadge = 'X-Ray Router → CHEST (Pneumonia)';
    else if (routed_to === 'musculoskeletal') workflowBadge = 'X-Ray Router → MUSCULOSKELETAL (MURA)';
    else if (isUnknown) workflowBadge = 'X-Ray Router → UNKNOWN';
  } else if (modality === 'ct') {
    if (routed_to === 'ct_brain_bleed') workflowBadge = 'CT Scan → Cranial Brain Bleed Classifier (CQ500/RSNA)';
    else workflowBadge = 'CT Scan → Thoracic 2D Axial Slice Classifier';
  } else if (modality === 'mri') {
    if (result.metrics?.['Cascade Finding'] || result.metrics?.['Stage 1 Tumor Screener']) {
      workflowBadge = 'Cascade Brain MRI: Stage 1 Tumor Normal → Stage 2 Dementia Reflex';
    } else if (routed_to === 'mri_dementia') {
      workflowBadge = 'MRI → Neurodegenerative Dementia Staging (ADNI/OASIS)';
    } else {
      workflowBadge = 'MRI → Brain Tumor Pathology Classifier (BraTS)';
    }
  }

  return (
    <div
      className={`glass-panel rounded-2xl p-6 sm:p-8 border text-left relative overflow-hidden transition-all ${
        isUnknown
          ? 'border-amber-500/40 bg-amber-950/20'
          : isPositiveFinding
          ? 'border-red-500/50 bg-slate-900/90 shadow-[0_0_40px_rgba(239,68,68,0.22)]'
          : 'border-emerald-500/40 bg-slate-900/80 shadow-[0_0_35px_rgba(16,185,129,0.15)]'
      }`}
    >
      {/* Background ambient glow */}
      <div
        className={`absolute top-0 right-0 w-64 h-64 rounded-full blur-3xl pointer-events-none ${
          isPositiveFinding ? 'bg-red-500/10' : 'bg-cyan-500/5'
        }`}
      />

      {/* Top Header / Router Breadcrumb */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <div className="flex items-center gap-2">
          <span
            className={`text-xs font-mono uppercase tracking-wider font-semibold px-2.5 py-1 rounded ${
              isPositiveFinding
                ? 'text-red-400 bg-red-950/70 border border-red-500/30'
                : 'text-cyan-400 bg-cyan-950/70 border border-cyan-500/20'
            }`}
          >
            {workflowBadge}
          </span>
          {isPositiveFinding && (
            <span className="flex items-center gap-1 text-[10px] font-mono font-bold uppercase tracking-wider text-red-400 bg-red-500/15 border border-red-500/30 px-2 py-0.5 rounded-full animate-pulse">
              <AlertCircle className="w-3 h-3 text-red-400" />
              <span>Positive Finding</span>
            </span>
          )}
        </div>
        <div className="text-[11px] font-mono text-slate-400">
          Threshold: <span className="text-slate-200">{(threshold * 100).toFixed(0)}%</span>
        </div>
      </div>

      {isUnknown ? (
        /* Unknown Router State */
        <div className="space-y-3">
          <div className="flex items-center gap-3 text-amber-400">
            <AlertOctagon className="w-8 h-8 shrink-0" />
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-white font-heading">
                UNABLE TO CONFIDENTLY ROUTE SCAN
              </h2>
              <p className="text-xs text-amber-300/90 mt-0.5">
                The router could not match this image to Chest or Musculoskeletal benchmarks with sufficient confidence.
              </p>
            </div>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed bg-black/40 p-3 rounded-xl border border-amber-500/20">
            Safety Mechanism: To protect clinical evaluation integrity, no specialist classification model was executed. Please verify anatomical orientation and upload an explicit Chest or Musculoskeletal radiograph.
          </p>
        </div>
      ) : (
        /* Definitive Prediction Display */
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 items-center">
            <div>
              <span className="text-xs font-mono uppercase tracking-wider text-slate-400 block mb-1">
                {result.metrics?.['Cascade Finding'] ? 'Consolidated Cascade Finding' : 'Primary Model Prediction'}
              </span>
              <div className="flex items-baseline gap-3">
                <h2
                  className={`text-3xl sm:text-4xl font-extrabold tracking-tight font-heading ${
                    isPositiveFinding
                      ? 'text-red-500 drop-shadow-[0_0_20px_rgba(239,68,68,0.7)]'
                      : 'text-emerald-400'
                  }`}
                >
                  {prediction}
                </h2>
              </div>
              <p className="text-xs text-slate-400 mt-2">
                {isPositiveFinding
                  ? 'Anatomical biomarkers and feature heatmaps indicate significant classification confidence.'
                  : 'No characteristic pathological biomarkers detected above decision threshold.'}
              </p>
            </div>

            {/* Confidence Score Pill */}
            <div className="sm:border-l sm:border-white/10 sm:pl-6 space-y-2">
              <span className="text-xs font-mono uppercase tracking-wider text-slate-400 block">
                Confidence Score
              </span>
              <div className="flex items-baseline gap-2">
                <span
                  className={`text-4xl sm:text-5xl font-mono font-bold tracking-tight ${
                    isPositiveFinding ? 'text-red-400' : 'text-white'
                  }`}
                >
                  {confidencePct}%
                </span>
                <span className={`text-xs font-mono ${isPositiveFinding ? 'text-red-400' : 'text-cyan-400'}`}>
                  P(Class)
                </span>
              </div>
              <div className="w-full bg-black/50 h-2 rounded-full overflow-hidden border border-white/5">
                <div
                  className={`h-full rounded-full transition-all duration-1000 ${
                    isPositiveFinding
                      ? 'bg-gradient-to-r from-red-500 to-rose-600 shadow-[0_0_12px_#ef4444]'
                      : 'bg-emerald-400'
                  }`}
                  style={{ width: `${confidencePct}%` }}
                />
              </div>
            </div>
          </div>

          {/* Cascade Brain MRI Two-Stage Diagnostic Breakdown (matching gui_predict) */}
          {result.metrics?.['Stage 1 Tumor Screener'] && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-4 border-t border-white/10 text-xs font-mono">
              <div className="p-3.5 rounded-xl bg-slate-950/80 border border-emerald-500/30 space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">
                    Stage 1: Brain Tumor Screening
                  </span>
                  <span className="px-1.5 py-0.5 rounded text-[9px] bg-emerald-950 text-emerald-400 border border-emerald-500/30">
                    Normal
                  </span>
                </div>
                <div className="text-emerald-400 font-bold text-sm">
                  NO BRAIN TUMOR (NORMAL BRAIN)
                </div>
                <p className="text-[11px] text-slate-400">
                  {result.metrics['Stage 1 Tumor Screener']}
                </p>
              </div>

              <div className={`p-3.5 rounded-xl bg-slate-950/80 border space-y-1 ${
                isPositiveFinding ? 'border-red-500/40' : 'border-emerald-500/30'
              }`}>
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-purple-400 uppercase tracking-wider font-bold">
                    Stage 2: Reflex Dementia Assessment
                  </span>
                  <span className="px-1.5 py-0.5 rounded text-[9px] bg-sky-950 text-sky-400 border border-sky-500/30">
                    Triggered
                  </span>
                </div>
                <div className={`font-bold text-sm ${isPositiveFinding ? 'text-red-400' : 'text-emerald-400'}`}>
                  {prediction}
                </div>
                <p className="text-[11px] text-slate-400">
                  Confidence: {confidencePct}% (Calibrated)
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Warnings Banner (Only rendered if warnings exist) */}
      {warnings && warnings.length > 0 && (
        <div className="mt-6 p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-200 text-xs space-y-1">
          <div className="flex items-center gap-1.5 font-semibold text-amber-300 font-mono">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>Operational Warnings</span>
          </div>
          <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px]">
            {warnings.map((warn, i) => (
              <li key={i}>{warn}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
