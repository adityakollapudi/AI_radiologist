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
    ct: 'Single 2D Axial Slice (Normal / Pneumonia / COVID-19)',
    mri: 'Intracranial T1ce/FLAIR Axial Sequence (Tumor / No Tumor)',
  };

  const handlePresetSelect = (preset) => {
    setSelectedSubtype(preset.subtype);
    // Generate image preview for preset
    onFileSelected(preset.id, preset.label);
  };

  const handleCustomFile = (file) => {
    setSelectedSubtype('default');
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
          <AnalysisLoader phase={analysisPhase} previewUrl={previewUrl} />
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
