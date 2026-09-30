import React, { useState } from 'react';
import { ArrowRight, Activity } from 'lucide-react';

import PhotorealisticAnatomy from '../components/three/PhotorealisticAnatomy';

export default function Home({ onStartAnalysis, onSelectModality }) {
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [hoveredRegion, setHoveredRegion] = useState(null);

  const handleStart = () => {
    setIsTransitioning(true);
    // Smooth transition sequence: animate anatomy, then trigger screen navigation
    setTimeout(() => {
      onStartAnalysis();
    }, 450);
  };

  const getRegionName = (key) => {
    switch (key) {
      case 'skull': return 'CRANIAL CAVITY & SKULL (MRI TARGET)';
      case 'chest': return 'THORACIC CAGE & LUNGS (X-RAY / CT TARGET)';
      case 'spine': return 'VERTEBRAL COLUMN & SKELETON (MURA TARGET)';
      case 'abdomen': return 'ABDOMINAL VISCERA';
      case 'pelvis': return 'PELVIC GIRDLE';
      default: return null;
    }
  };

  return (
    <div className="relative w-full min-h-[calc(100vh-65px)] flex items-center justify-center overflow-hidden">
      {/* Background medical grid pattern & radial vignettes */}
      <div className="absolute inset-0 bg-[#030712] pointer-events-none" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(6,182,212,0.12)_0%,rgba(3,7,18,0.95)_75%)] pointer-events-none" />
      <div className="absolute inset-0 scan-line-overlay opacity-25 pointer-events-none" />

      {/* Main Grid: Responsive composition matching user's architecture diagram */}
      <div className="container mx-auto px-6 sm:px-12 py-10 relative z-10 w-full flex flex-col lg:flex-row items-center justify-between gap-8 lg:gap-12 min-h-[80vh]">
        {/* Left Column: Hero Text & Dominant CTA */}
        <div
          className={`flex-1 text-left space-y-6 max-w-xl transition-all duration-500 ${
            isTransitioning ? 'opacity-0 -translate-x-12' : 'opacity-100 translate-x-0'
          }`}
        >
          {/* Status Indicator */}
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/70 border border-cyan-500/30 text-cyan-300 text-xs font-mono tracking-wider">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
            <span>NEURAL RADIOLOGY SUITE • v2.4</span>
          </div>

          {/* Heading */}
          <div className="space-y-2">
            <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-white font-heading uppercase leading-[1.08]">
              A UNIFIED <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-300 to-blue-500 medical-glow-cyan">
                AI RADIOLOGIST
              </span>
            </h1>
            <p className="text-base sm:text-lg text-slate-300 font-normal leading-relaxed pt-2">
              AI-assisted analysis across X-ray, CT, and MRI.
            </p>
          </div>

          {/* Primary Action Button (DOMINANT CTA) */}
          <div className="pt-4 flex flex-wrap items-center gap-4">
            <button
              type="button"
              onClick={handleStart}
              className="px-8 py-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-extrabold text-sm uppercase tracking-wider shadow-[0_0_30px_rgba(6,182,212,0.45)] hover:shadow-[0_0_45px_rgba(6,182,212,0.65)] hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center gap-3 cursor-pointer group"
            >
              <span>START ANALYSIS</span>
              <ArrowRight className="w-4 h-4 text-slate-950 group-hover:translate-x-1 transition-transform" />
            </button>

            <button
              type="button"
              onClick={() => {
                const el = document.getElementById('anatomy-viewport');
                el?.scrollIntoView({ behavior: 'smooth' });
              }}
              className="px-5 py-3.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-slate-300 hover:text-white text-xs font-mono tracking-wider uppercase transition-all"
            >
              Explore 3D Hologram
            </button>
          </div>

          {/* Hologram Region HUD Indicator on Hover */}
          {hoveredRegion && (
            <div className="inline-flex items-center gap-2 p-2.5 px-3.5 rounded-lg bg-cyan-950/80 border border-cyan-400/40 text-cyan-300 font-mono text-xs animate-fadeIn">
              <Activity className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
              <span>INSPECTING: {getRegionName(hoveredRegion)}</span>
            </div>
          )}

          {/* Minimal capabilities tag strip */}
          <div className="pt-6 border-t border-white/10 flex items-center gap-6 text-xs font-mono text-slate-400">
            <div
              onClick={() => onSelectModality && onSelectModality('xray')}
              className="cursor-pointer hover:text-cyan-300 transition-colors"
            >
              <span className="text-white font-semibold">01</span> X-Ray (Chest/MURA)
            </div>
            <div
              onClick={() => onSelectModality && onSelectModality('ct')}
              className="cursor-pointer hover:text-cyan-300 transition-colors"
            >
              <span className="text-white font-semibold">02</span> CT (Axial Slices)
            </div>
            <div
              onClick={() => onSelectModality && onSelectModality('mri')}
              className="cursor-pointer hover:text-cyan-300 transition-colors"
            >
              <span className="text-white font-semibold">03</span> Brain MRI
            </div>
          </div>
        </div>

        {/* Right Column: Dedicated Photorealistic Holographic Anatomy Viewport */}
        <div
          id="anatomy-viewport"
          className="flex-1 w-full h-[520px] sm:h-[620px] lg:h-[740px] relative flex flex-col items-center justify-center"
        >
          <div className="w-full h-full relative">
            <PhotorealisticAnatomy
              isTransitioning={isTransitioning}
              onHoverRegion={setHoveredRegion}
              onSelectModality={onSelectModality}
              activeRegion={hoveredRegion}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
