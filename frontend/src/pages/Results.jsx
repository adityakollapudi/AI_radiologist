import React, { useState } from 'react';

import { ArrowLeft, Download, RotateCcw, Activity, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import { downloadPdfReport } from '../services/api';
import ResultCard from '../components/results/ResultCard';
import ProbabilityBars from '../components/results/ProbabilityBars';
import ScanViewer from '../components/results/ScanViewer';
import ModelDetails from '../components/results/ModelDetails';
import Disclaimer from '../components/results/Disclaimer';

export default function Results({ result, onAnalyzeAnother, onBackToSelection }) {
  const sessionId = '94821';
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState(null);
  const [downloadSuccess, setDownloadSuccess] = useState(false);

  if (!result) {
    return (
      <div className="container mx-auto px-4 py-20 text-center space-y-4">
        <p className="text-slate-400 font-mono text-sm">No active analysis session found.</p>
        <button
          onClick={onBackToSelection}
          className="px-6 py-2.5 rounded-xl bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 text-xs font-mono uppercase"
        >
          Select a Scan to Analyze
        </button>
      </div>
    );
  }

  const handleDownloadPdf = async () => {
    if (isDownloading) return;
    setIsDownloading(true);
    setDownloadError(null);
    setDownloadSuccess(false);

    try {
      await downloadPdfReport(result);
      setDownloadSuccess(true);
      setTimeout(() => {
        setDownloadSuccess(false);
      }, 4000);
    } catch (err) {
      console.error('PDF Report generation error:', err);
      setDownloadError(err.message || 'Failed to generate PDF report.');
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="relative w-full min-h-[calc(100vh-65px)] px-4 sm:px-8 py-8 sm:py-10">
      <div className="absolute inset-0 bg-[#030712] pointer-events-none" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(6,182,212,0.06)_0%,rgba(3,7,18,0.98)_70%)] pointer-events-none" />
      <div className="absolute inset-0 scan-line-overlay opacity-20 pointer-events-none" />

      <div className="container mx-auto max-w-5xl relative z-10 w-full space-y-8">
        {/* Top Header Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-5">
          <div className="flex items-center gap-3 text-left">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Activity className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl sm:text-2xl font-extrabold text-white font-heading tracking-tight">
                  ANALYSIS COMPLETE
                </h1>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-950 border border-cyan-500/30 text-cyan-400 uppercase">
                  Session ID: #{sessionId}
                </span>
              </div>

              <p className="text-xs text-slate-400 font-mono">
                Multimodal AI inference, probability calibration, and explainability map verified
              </p>
            </div>
          </div>

          {/* Action CTAs */}
          <div className="flex flex-col sm:flex-row items-end sm:items-center gap-3">
            <button
              type="button"
              onClick={handleDownloadPdf}
              disabled={isDownloading}
              className="px-4 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-slate-200 hover:text-white text-xs font-mono tracking-wider flex items-center gap-2 transition-all cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isDownloading ? (
                <Loader2 className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
              ) : (
                <Download className="w-3.5 h-3.5 text-cyan-400" />
              )}
              <span>{isDownloading ? 'GENERATING PDF...' : 'DOWNLOAD PDF REPORT'}</span>
            </button>

            <button
              type="button"
              onClick={onAnalyzeAnother}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 text-xs font-bold font-mono tracking-wider flex items-center gap-2 transition-all cursor-pointer shadow-[0_0_20px_rgba(6,182,212,0.3)]"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>ANALYZE ANOTHER SCAN</span>
            </button>
          </div>
        </div>

        {/* Download Feedback Alerts */}
        {downloadError && (
          <div className="flex items-center gap-2.5 text-xs font-mono text-rose-300 bg-rose-950/40 border border-rose-500/30 px-4 py-2.5 rounded-xl">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{downloadError}</span>
          </div>
        )}

        {downloadSuccess && (
          <div className="flex items-center gap-2.5 text-xs font-mono text-emerald-300 bg-emerald-950/40 border border-emerald-500/30 px-4 py-2.5 rounded-xl">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>Medical PDF report successfully generated and downloaded.</span>
          </div>
        )}

        {/* 1. Large Result Card (Primary Prediction + Confidence Score) */}
        <ResultCard result={result} />

        {/* 2. Primary Medical Scan & Grad-CAM Visualization */}
        <ScanViewer
          originalScanUrl={result.original_png_base64}
          gradcamScanUrl={result.gradcam_png_base64}
          modality={result.modality}
          visualizations={result.visualizations || result.metadata?.visualizations}
          prediction={result.prediction}
          metadata={result.metadata}
        />

        {/* 3. Class Probability Softmax Bars */}
        <ProbabilityBars
          probabilities={result.probabilities}
          predictedClass={result.prediction}
        />

        {/* 4. Expandable Verified Model Details */}
        <ModelDetails result={result} />

        {/* 5. Mandatory Medical Disclaimer (Always visible, not hidden in modal) */}
        <Disclaimer />

        {/* Bottom Navigation */}
        <div className="pt-4 flex items-center justify-between text-xs font-mono text-slate-500">
          <button
            onClick={onBackToSelection}
            className="hover:text-slate-300 transition-colors flex items-center gap-1.5"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Return to Modality Selection</span>
          </button>
          <span>MedVision AI Clinical Research Suite</span>
        </div>
      </div>
    </div>
  );
}
