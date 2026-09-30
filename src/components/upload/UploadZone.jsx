import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Sparkles, AlertCircle } from 'lucide-react';
import { SAMPLE_PRESETS } from '../../utils/mockData';

export default function UploadZone({
  modality,
  onFileSelected,
  onPresetSelected,
  _hasFile,
  disabled = false,
}) {

  const [isDragOver, setIsDragOver] = useState(false);
  const [dragError, setDragError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    if (disabled) return;
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      validateAndPassFile(file);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      validateAndPassFile(file);
    }
  };

  const validateAndPassFile = (file) => {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
      setDragError('Invalid format. Only JPG, JPEG, and PNG images are supported.');
      return;
    }
    setDragError(null);
    onFileSelected(file);
  };

  // Filter sample presets for current modality
  const filteredPresets = SAMPLE_PRESETS.filter((p) => p.modality === modality);

  return (
    <div className="w-full space-y-4">
      {/* Interactive Drop Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`relative w-full rounded-2xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-300 overflow-hidden ${
          isDragOver
            ? 'border-2 border-cyan-400 bg-cyan-950/30 scale-[1.01] shadow-[0_0_40px_rgba(6,182,212,0.35)]'
            : 'border border-cyan-500/25 bg-slate-950/60 hover:border-cyan-500/50 hover:bg-slate-900/50 shadow-2xl'
        }`}
      >
        {/* Animated Laser Scanning Beam on Drag */}
        {isDragOver && (
          <div className="absolute inset-0 pointer-events-none">
            <div className="w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_15px_#22d3ee] animate-scan-sweep" />
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,image/jpeg,image/png"
          onChange={handleFileInputChange}
          disabled={disabled}
          className="hidden"
          aria-label="Upload medical scan"
        />

        <div className="flex flex-col items-center justify-center space-y-4 relative z-10">
          <div
            className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-all ${
              isDragOver
                ? 'bg-cyan-500/20 text-cyan-300 shadow-[0_0_25px_rgba(6,182,212,0.5)]'
                : 'bg-white/5 border border-white/10 text-cyan-400 group-hover:scale-105'
            }`}
          >
            <UploadCloud className="w-8 h-8" />
          </div>

          <div>
            <p className="text-base sm:text-lg font-medium text-white font-heading">
              Drop your scan here
            </p>
            <p className="text-xs text-slate-400 mt-1">
              or click to browse from your device
            </p>
          </div>

          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-[11px] font-mono text-slate-300">
            <span>Supported:</span>
            <span className="text-cyan-400 font-semibold">JPG</span>
            <span className="text-slate-500">•</span>
            <span className="text-cyan-400 font-semibold">JPEG</span>
            <span className="text-slate-500">•</span>
            <span className="text-cyan-400 font-semibold">PNG</span>
          </div>

          <button
            type="button"
            className="mt-2 px-5 py-2 rounded-xl bg-cyan-500/15 hover:bg-cyan-500/25 border border-cyan-500/30 text-cyan-300 text-xs font-semibold tracking-wider uppercase transition-all"
          >
            Select Scan
          </button>
        </div>
      </div>

      {dragError && (
        <div className="flex items-center gap-2 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs text-left">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{dragError}</span>
        </div>
      )}

      {/* Preset Medical Scans for instant testing */}
      {filteredPresets.length > 0 && (
        <div className="pt-2 text-left">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
              Verified Sample Presets
            </span>
            <span className="text-[10px] text-slate-500">1-click test scans</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {filteredPresets.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => onPresetSelected(preset)}
                className="p-2.5 rounded-xl bg-slate-900/60 border border-white/5 hover:border-cyan-500/40 hover:bg-slate-900/90 text-left transition-all flex items-start gap-2.5 group"
              >
                <div className="p-1.5 rounded-lg bg-white/5 text-cyan-400 group-hover:text-cyan-300 mt-0.5">
                  <ImageIcon className="w-3.5 h-3.5" />
                </div>
                <div>
                  <div className="text-xs font-medium text-slate-200 group-hover:text-white">
                    {preset.label}
                  </div>
                  <div className="text-[10px] text-slate-400 line-clamp-1">
                    {preset.description}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
