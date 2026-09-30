import React, { useState } from 'react';
import { Activity, Cpu, Info, Sun, Moon } from 'lucide-react';
import ModelsModal from './ModelsModal';

export default function Navbar({ currentStep, onNavigate }) {
  const [showModels, setShowModels] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [isHighContrast, setIsHighContrast] = useState(false);

  const toggleTheme = () => {
    setIsHighContrast((prev) => {
      const next = !prev;
      document.documentElement.classList.toggle('high-contrast-mode', next);
      return next;
    });
  };

  return (
    <>
      <header className="fixed top-0 left-0 right-0 z-40 px-4 sm:px-8 py-3.5 flex items-center justify-between border-b border-white/5 bg-[#030712]/75 backdrop-blur-xl">
        {/* Left: Brand */}
        <button
          onClick={() => onNavigate('home')}
          className="flex items-center gap-2.5 text-left group transition-transform focus:outline-none"
          aria-label="MedVision AI Home"
        >
          <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:border-cyan-400/60 transition-all">
            <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
          </div>
          <div>
            <div className="text-base font-bold tracking-wider text-white font-heading flex items-center gap-1.5">
              <span>MEDVISION</span>
              <span className="text-cyan-400 font-mono text-xs px-1.5 py-0.2 rounded bg-cyan-950/60 border border-cyan-500/30">AI</span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono hidden sm:block">A Unified AI Radiologist</p>
          </div>
        </button>

        {/* Center / Navigation Links */}
        <nav className="flex items-center gap-1 sm:gap-2">
          <button
            onClick={() => onNavigate('select')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              currentStep === 'select' || currentStep === 'upload' || currentStep === 'results'
                ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
            }`}
          >
            Analysis
          </button>

          <button
            onClick={() => setShowModels(true)}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-100 hover:bg-white/5 transition-all flex items-center gap-1.5"
          >
            <Cpu className="w-3.5 h-3.5 text-cyan-400/80" />
            <span>Models</span>
          </button>

          <button
            onClick={() => setShowAbout(true)}
            className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-100 hover:bg-white/5 transition-all flex items-center gap-1.5"
          >
            <Info className="w-3.5 h-3.5 text-slate-400" />
            <span className="hidden sm:inline">About</span>
          </button>

          {/* Theme / Contrast Toggle */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle contrast theme"
            className="ml-2 p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-white/5 border border-white/5 transition-colors"
            title={isHighContrast ? 'Standard Medical Dark' : 'High Contrast Medical Mode'}
          >
            {isHighContrast ? (
              <Sun className="w-4 h-4 text-cyan-400" />
            ) : (
              <Moon className="w-4 h-4 text-slate-400" />
            )}
          </button>
        </nav>
      </header>

      {/* Models Modal */}
      <ModelsModal isOpen={showModels} onClose={() => setShowModels(false)} />

      {/* About Modal */}
      {showAbout && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md"
          onClick={() => setShowAbout(false)}
        >
          <div
            className="glass-panel w-full max-w-lg rounded-2xl p-6 border border-cyan-500/30 text-left relative"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-lg font-bold text-white font-heading mb-2">About MedVision AI</h2>
            <p className="text-xs text-slate-300 leading-relaxed mb-4">
              MedVision AI unifies three independent machine learning radiology models into a cohesive,
              interactive medical imaging workstation for research and educational purposes.
            </p>
            <div className="space-y-2 text-xs font-mono text-slate-400 bg-black/40 p-3 rounded-lg border border-white/5">
              <p>• X-Ray: Chest Pneumonia & MURA Musculoskeletal Abnormality</p>
              <p>• CT Scan: Normal, Pneumonia, and COVID-19 2D Axial Slices</p>
              <p>• MRI: Brain Tumor vs. No Tumor Analysis with Grad-CAM</p>
            </div>
            <div className="mt-5 text-right">
              <button
                onClick={() => setShowAbout(false)}
                className="px-4 py-1.5 rounded-lg bg-cyan-500/20 text-cyan-300 text-xs font-medium border border-cyan-500/30 hover:bg-cyan-500/30 transition-all"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
