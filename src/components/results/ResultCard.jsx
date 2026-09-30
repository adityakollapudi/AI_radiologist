import React from 'react';
import { AlertTriangle, AlertOctagon } from 'lucide-react';


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
  const isPositiveFinding =
    prediction === 'PNEUMONIA' ||
    prediction === 'ABNORMAL' ||
    prediction === 'COVID-19' ||
    prediction === 'TUMOR';

  const confidencePct = (confidence * 100).toFixed(1);

  // Modality & Routing Title
  let workflowBadge = 'Standard Pipeline';
  if (modality === 'xray') {
    if (routed_to === 'chest') workflowBadge = 'X-Ray Router → CHEST (Pneumonia)';
    else if (routed_to === 'musculoskeletal') workflowBadge = 'X-Ray Router → MUSCULOSKELETAL (MURA)';
    else if (isUnknown) workflowBadge = 'X-Ray Router → UNKNOWN';
  } else if (modality === 'ct') {
    workflowBadge = 'CT Scan → 2D Axial Slice Classifier';
  } else if (modality === 'mri') {
    workflowBadge = 'MRI → Brain Pathology Classifier';
  }

  return (
    <div
      className={`glass-panel rounded-2xl p-6 sm:p-8 border text-left relative overflow-hidden transition-all ${
        isUnknown
          ? 'border-amber-500/40 bg-amber-950/20'
          : isPositiveFinding
          ? 'border-cyan-500/40 bg-slate-900/80 shadow-[0_0_35px_rgba(6,182,212,0.15)]'
          : 'border-emerald-500/40 bg-slate-900/80'
      }`}
    >
      {/* Background ambient glow */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />

      {/* Top Header / Router Breadcrumb */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <span className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-semibold px-2.5 py-1 rounded bg-cyan-950/70 border border-cyan-500/20">
          {workflowBadge}
        </span>
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
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 items-center">
          <div>
            <span className="text-xs font-mono uppercase tracking-wider text-slate-400 block mb-1">
              Primary Model Prediction
            </span>
            <div className="flex items-baseline gap-3">
              <h2
                className={`text-3xl sm:text-4xl font-extrabold tracking-tight font-heading ${
                  isPositiveFinding ? 'text-cyan-300 medical-glow-cyan' : 'text-emerald-400'
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
              <span className="text-4xl sm:text-5xl font-mono font-bold text-white tracking-tight">
                {confidencePct}%
              </span>
              <span className="text-xs font-mono text-cyan-400">P(Class)</span>
            </div>
            <div className="w-full bg-black/50 h-2 rounded-full overflow-hidden border border-white/5">
              <div
                className={`h-full rounded-full transition-all duration-1000 ${
                  isPositiveFinding
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-500 shadow-[0_0_10px_#06b6d4]'
                    : 'bg-emerald-400'
                }`}
                style={{ width: `${confidencePct}%` }}
              />
            </div>
          </div>
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
