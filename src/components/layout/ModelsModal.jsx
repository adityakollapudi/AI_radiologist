import React from 'react';
import { X, Cpu, ShieldAlert } from 'lucide-react';


export default function ModelsModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="models-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md"
      onClick={onClose}
    >
      <div
        className="glass-panel w-full max-w-3xl rounded-2xl p-6 sm:p-8 border border-cyan-500/30 text-left relative max-h-[85vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          aria-label="Close Models Modal"
          className="absolute top-5 right-5 p-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-3 mb-2">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h2 id="models-modal-title" className="text-xl sm:text-2xl font-bold tracking-tight text-white font-heading">
              Verified Model Architecture
            </h2>
            <p className="text-xs sm:text-sm text-slate-400">
              Technical specifications of the three integrated deep learning pipelines
            </p>
          </div>
        </div>

        <div className="space-x-0 space-y-4 mt-6">
          {/* 1. X-RAY */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-white/10 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-cyan-400 font-mono">
                Modality 01 • X-Ray Radiography
              </span>
              <span className="text-[11px] px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-mono">
                Router Pipeline
              </span>
            </div>
            <h3 className="text-base font-semibold text-white">Chest & Musculoskeletal Router</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Multi-stage pipeline: First routes anatomical view to <strong>Chest</strong> or <strong>Musculoskeletal (MURA)</strong> or <strong>Unknown</strong>.
              Chest images are evaluated with <em>EfficientNet-B3</em> for Pneumonia. Musculoskeletal images are evaluated with <em>DenseNet-169</em> for structural abnormalities.
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 text-xs font-mono">
              <div className="p-2 rounded bg-white/5">
                <span className="text-slate-400 block text-[10px]">Chest Input</span>
                224×224 px
              </div>
              <div className="p-2 rounded bg-white/5">
                <span className="text-slate-400 block text-[10px]">MURA Input</span>
                320×320 px
              </div>
              <div className="p-2 rounded bg-white/5">
                <span className="text-slate-400 block text-[10px]">Chest AUC</span>
                0.932
              </div>
              <div className="p-2 rounded bg-white/5">
                <span className="text-slate-400 block text-[10px]">MURA Kappa</span>
                0.741
              </div>
            </div>
          </div>

          {/* 2. CT SCAN */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-white/10 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-blue-400 font-mono">
                Modality 02 • Computed Tomography
              </span>
              <span className="text-[11px] px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-300 border border-blue-500/20 font-mono">
                2D Axial Slice
              </span>
            </div>
            <h3 className="text-base font-semibold text-white">Normal / Pneumonia / COVID-19 Classifier</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Processes individual 2D axial thoracic CT slices using <em>DenseNet-121</em> trained on the CC-CCII & COVID-CT cohort.
              Identifies peripheral bilateral ground-glass opacities and consolidations.
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 text-xs font-mono">
              <div className="p-2 rounded bg-white/5">
                <span className="text-slate-400 block text-[10px]">Input Size</span>
                512×512 px
              </div>
              <div className="p-2 rounded bg-white/5">
                <span className="text-slate-400 block text-[10px]">Threshold</span>
                0.65
              </div>
              <div className="p-2 rounded bg-white/5">
                <span className="text-slate-400 block text-[10px]">Sensitivity</span>
                87.5%
              </div>
              <div className="p-2 rounded bg-white/5">
                <span className="text-slate-400 block text-[10px]">Specificity</span>
                91.0%
              </div>
            </div>
          </div>

          {/* 3. MRI */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-white/10 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-purple-400 font-mono">
                Modality 03 • Magnetic Resonance Imaging
              </span>
              <span className="text-[11px] px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-300 border border-purple-500/20 font-mono">
                Brain Pathology
              </span>
            </div>
            <h3 className="text-base font-semibold text-white">Brain Tumor / No Tumor Classifier</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Evaluates axial T1ce/FLAIR contrast sequences for intracranial neoplasms using <em>ResNet-50</em> fine-tuned on the BraTS challenge repository.
              Produces class likelihood and Grad-CAM localized attention.
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 text-xs font-mono">
              <div className="p-2 rounded bg-white/5">
                <span className="text-slate-400 block text-[10px]">Input Size</span>
                224×224 px
              </div>
              <div className="p-2 rounded bg-white/5">
                <span className="text-slate-400 block text-[10px]">Threshold</span>
                0.70
              </div>
              <div className="p-2 rounded bg-white/5">
                <span className="text-slate-400 block text-[10px]">ROC-AUC</span>
                0.963
              </div>
              <div className="p-2 rounded bg-white/5">
                <span className="text-slate-400 block text-[10px]">Accuracy</span>
                93.8%
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-white/10 flex items-center justify-between">
          <p className="text-[11px] text-slate-400 flex items-center gap-1.5">
            <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
            Research & Educational verification only.
          </p>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 text-xs font-medium border border-cyan-500/30 transition-all"
          >
            Close Overview
          </button>
        </div>
      </div>
    </div>
  );
}
