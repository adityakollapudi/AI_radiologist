import React, { useState } from 'react';
import { ArrowLeft, AlertCircle } from 'lucide-react';

import UploadZone from '../components/upload/UploadZone';
import FilePreview from '../components/upload/FilePreview';
import AnalysisLoader from '../components/analysis/AnalysisLoader';

export default function Upload({
  modality,
  previewUrl,
  currentFile,
  isAnalyzing,
  analysisPhase,
  error,
  onFileSelected,
  onAnalyze,
  onChangeScan,
  onBack,
}) {
  const [selectedSubtype, setSelectedSubtype] = useState('default');

  const modalityNames = {
    xray: 'X-RAY ANALYSIS',
    ct: 'CT SCAN ANALYSIS',
    mri: 'BRAIN MRI ANALYSIS',
  };

  const modalitySubtitles = {
    xray: 'Chest Pneumonia & Musculoskeletal (MURA) Detection',
    ct: 'Thoracic Slices (Normal / Pneumonia / COVID-19) & Cranial CT (Brain Bleed)',
    mri: 'Brain Neoplasms (Tumor / No Tumor) & Dementia Atrophy Analysis',
  };

  const pipelineOptions = {
    xray: [
      { id: 'default', label: 'Chest Pneumonia', badge: 'NIH ChestX-ray14' },
      { id: 'musculoskeletal', label: 'Musculoskeletal (MURA)', badge: 'Stanford MURA' },
    ],
    ct: [
      { id: 'default', label: 'Thoracic (COVID-19 / Pneumonia)', badge: 'CC-CCII Cohort' },
      { id: 'brain_bleed', label: 'Cranial Brain Bleed (Hemorrhage)', badge: 'CQ500 / RSNA' },
    ],
    mri: [
      { id: 'default', label: 'Brain Tumor (Tumor / No Tumor)', badge: 'BraTS Benchmark' },
      { id: 'dementia', label: 'Dementia Staging (Atrophy)', badge: 'ADNI / OASIS' },
    ],
  };

  const currentPipelines = pipelineOptions[modality] || [];

  const handlePipelineSwitch = (subtypeId) => {
    setSelectedSubtype(subtypeId);
    if (typeof currentFile === 'string') {
      // Re-trigger preset generation for matching subtype if on preset scan
      if (modality === 'ct') {
        const targetId = subtypeId === 'brain_bleed' ? 'ct_brain_bleed' : 'ct_covid';
        onFileSelected(targetId, targetId, subtypeId);
      } else if (modality === 'mri') {
        const targetId = subtypeId === 'dementia' ? 'mri_dementia' : 'mri_tumor';
        onFileSelected(targetId, targetId, subtypeId);
      } else if (modality === 'xray') {
        const targetId = subtypeId === 'musculoskeletal' ? 'xray_mura_fracture' : 'xray_chest_pneumonia';
        onFileSelected(targetId, targetId, subtypeId);
      }
    }
  };

  const handlePresetSelect = (preset) => {
    setSelectedSubtype(preset.subtype);
    // Generate image preview for preset
    onFileSelected(preset.id, preset.label, preset.subtype);
  };

  const handleCustomFile = (file) => {
    onFileSelected(file);
  };

  return (
    <div className="relative w-full min-h-[calc(100vh-65px)] flex flex-col items-center justify-center px-4 sm:px-8 py-10">
      <div className="absolute inset-0 bg-[#030712] pointer-events-none" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(6,182,212,0.06)_0%,rgba(3,7,18,0.98)_85%)] pointer-events-none" />
      <div className="absolute inset-0 scan-line-overlay opacity-20 pointer-events-none" />

      <div className="container mx-auto max-w-3xl relative z-10 w-full space-y-8">
        {/* Top Bar with Back Button matching wireframe */}
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={onBack}
            disabled={isAnalyzing}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-mono text-slate-300 hover:text-white transition-all cursor-pointer"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span className="font-semibold uppercase tracking-wider">
              ← {modality ? modality.toUpperCase() : 'SELECTION'}
            </span>
          </button>

          <span className="text-[11px] font-mono text-cyan-400 uppercase tracking-wider">
            STEP 03 OF 04 • SCAN INGESTION
          </span>
        </div>

        {/* Dynamic Display: If actively analyzing, show the Medical AI Loader sequence */}
        {isAnalyzing ? (
          <AnalysisLoader phase={analysisPhase} previewUrl={previewUrl} modality={modality} />
        ) : (
          <>
            {/* Header matching wireframe */}
            <div className="text-center space-y-2">
              <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold px-2.5 py-1 rounded bg-cyan-950/60 border border-cyan-500/20 inline-block">
                {modalityNames[modality] || 'UPLOAD MEDICAL SCAN'}
              </span>
              <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white font-heading uppercase">
                UPLOAD MEDICAL SCAN
              </h2>
              <p className="text-xs sm:text-sm text-slate-400 font-mono">
                {modalitySubtitles[modality]}
              </p>
            </div>

            {/* Pipeline Target Selector Pill Tabs */}
            {currentPipelines.length > 0 && (
              <div className="flex flex-col items-center gap-2">
                <span className="text-[11px] font-mono uppercase text-slate-400 tracking-wider">
                  Target Diagnostic Pipeline:
                </span>
                <div className="inline-flex p-1 rounded-xl bg-slate-950/80 border border-white/10 shadow-lg">
                  {currentPipelines.map((pipe) => {
                    const isActive =
                      selectedSubtype === pipe.id ||
                      (pipe.id === 'default' && (selectedSubtype === 'normal' || selectedSubtype === 'default'));

                    return (
                      <button
                        key={pipe.id}
                        type="button"
                        onClick={() => handlePipelineSwitch(pipe.id)}
                        className={`px-3.5 py-1.5 rounded-lg text-xs font-mono transition-all flex items-center gap-2 cursor-pointer ${
                          isActive
                            ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-[0_0_15px_rgba(6,182,212,0.3)] font-semibold'
                            : 'text-slate-400 hover:text-slate-200 border border-transparent'
                        }`}
                      >
                        <span>{pipe.label}</span>
                        <span className="text-[10px] opacity-60 hidden sm:inline">({pipe.badge})</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {error && (
              <div className="flex items-center gap-2 p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs text-left">
                <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
                <span>{error}</span>
              </div>
            )}

            {/* If file/preview exists, show FilePreview with [ ANALYZE SCAN ] button */}
            {previewUrl ? (
              <FilePreview
                previewUrl={previewUrl}
                fileName={typeof currentFile === 'string' ? currentFile : currentFile?.name || 'Selected Medical Scan'}
                fileSize={currentFile?.size}
                modality={modality}
                onAnalyze={() => onAnalyze({ subtype: selectedSubtype })}
                onChangeScan={onChangeScan}
                isAnalyzing={isAnalyzing}
              />
            ) : (
              /* Otherwise, show interactive Drop Zone */
              <UploadZone
                modality={modality}
                onFileSelected={handleCustomFile}
                onPresetSelected={handlePresetSelect}
                hasFile={Boolean(previewUrl)}
                disabled={isAnalyzing}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}
