import React, { useState, Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { ArrowLeft, ArrowRight, Clock } from 'lucide-react';

import XRayScene from '../components/three/XRayScene';
import CTScene from '../components/three/CTScene';
import MRIAnatomy from '../components/three/MRIAnatomy';

export default function ScanSelection({ onSelectModality, onBack }) {
  const [hoveredCard, setHoveredCard] = useState(null);
  const [selectedId, setSelectedId] = useState(null);

  const modalities = [
    {
      id: 'xray',
      name: 'X-RAY',
      subtitle: 'Radiography',
      desc: 'Pneumonia + Musculoskeletal Analysis',
      details: 'Dual-specialist pipeline: Automated routing to Chest (Pneumonia) or MURA (Musculoskeletal Abnormality).',
      badge: 'Dual Router',
      render3D: (isHovered) => <XRayScene isHovered={isHovered} />,
      cameraPos: [0, 0, 4.2],
    },
    {
      id: 'ct',
      name: 'CT SCAN',
      subtitle: 'Computed Tomography',
      desc: 'Normal / Pneumonia / COVID-19',
      details: 'Evaluates individual 2D axial thoracic slices for ground-glass opacities, consolidations, or normal lung parenchyma.',
      badge: '2D Axial Slice',
      render3D: (isHovered) => <CTScene isHovered={isHovered} />,
      cameraPos: [0, 0, 4.2],
    },
    {
      id: 'mri',
      name: 'MRI',
      subtitle: 'Magnetic Resonance',
      desc: 'Brain Tumor Analysis',
      details: 'Binary classification of intracranial axial T1ce/FLAIR sequences: Tumor vs. No Tumor with Grad-CAM localization.',
      badge: 'Brain Pathology',
      render3D: (isHovered) => <MRIAnatomy isHovered={isHovered} />,
      cameraPos: [0, 0, 4.2],
    },
  ];

  const handleCardClick = (id) => {
    setSelectedId(id);
    // Smooth transition
    setTimeout(() => {
      onSelectModality(id);
    }, 300);
  };

  return (
    <div className="relative w-full min-h-[calc(100vh-65px)] flex flex-col items-center justify-center px-4 sm:px-8 py-10">
      {/* Background vignette & scan lines */}
      <div className="absolute inset-0 bg-[#030712] pointer-events-none" />
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(6,182,212,0.08)_0%,rgba(3,7,18,0.98)_80%)] pointer-events-none" />
      <div className="absolute inset-0 scan-line-overlay opacity-20 pointer-events-none" />

      <div className="container mx-auto max-w-6xl relative z-10 w-full space-y-10 text-center">
        {/* Back navigation */}
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={onBack}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-mono text-slate-300 hover:text-white transition-all"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>BACK TO OVERVIEW</span>
          </button>

          <span className="text-[11px] font-mono text-slate-500 uppercase tracking-wider hidden sm:block">
            STEP 02 OF 04 • SELECTION
          </span>
        </div>

        {/* Section Heading matching architecture wireframe */}
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-500/20 text-cyan-400 text-xs font-mono uppercase tracking-wider">
            Select Modality
          </div>
          <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight text-white font-heading uppercase">
            WHAT ARE YOU ANALYZING?
          </h2>
          <p className="text-xs sm:text-sm text-slate-400 max-w-md mx-auto">
            Choose the medical imaging modality to analyze. Each route connects to its dedicated deep learning pipeline.
          </p>
        </div>

        {/* Three Interactive 3D Modality Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8 pt-4">
          {modalities.map((item) => {
            const isHovered = hoveredCard === item.id;
            const isSelected = selectedId === item.id;
            const isFaded = selectedId && selectedId !== item.id;

            return (
              <div
                key={item.id}
                onMouseEnter={() => setHoveredCard(item.id)}
                onMouseLeave={() => setHoveredCard(null)}
                onClick={() => handleCardClick(item.id)}
                className={`group relative rounded-2xl p-6 glass-panel cursor-pointer transition-all duration-300 flex flex-col justify-between overflow-hidden text-left ${
                  isSelected
                    ? 'border-cyan-400 bg-cyan-950/40 shadow-[0_0_40px_rgba(6,182,212,0.4)] scale-[1.02]'
                    : isHovered
                    ? 'border-cyan-500/60 -translate-y-2 shadow-[0_0_35px_rgba(6,182,212,0.25)]'
                    : 'border-white/10 hover:border-cyan-500/40'
                } ${isFaded ? 'opacity-30 scale-95' : 'opacity-100'}`}
              >
                {/* Modality Badge */}
                <div className="flex items-center justify-between relative z-10 mb-2">
                  <span className="text-[11px] font-mono uppercase px-2.5 py-1 rounded bg-black/60 border border-white/10 text-cyan-400 font-semibold">
                    {item.badge}
                  </span>
                  <span className="text-[10px] font-mono text-slate-500 uppercase">
                    {item.subtitle}
                  </span>
                </div>

                {/* 3D Canvas Stage */}
                <div className="w-full h-52 sm:h-56 relative my-2 rounded-xl overflow-hidden bg-slate-950/80 border border-white/5 flex items-center justify-center group-hover:border-cyan-500/30 transition-colors">
                  <Canvas
                    camera={{ position: item.cameraPos, fov: 42 }}
                    gl={{ antialias: true, alpha: true }}
                    className="w-full h-full"
                  >
                    <ambientLight intensity={0.7} />
                    <pointLight position={[4, 4, 4]} intensity={1.2} color="#00e5ff" />
                    <pointLight position={[-4, -4, -4]} intensity={0.6} color="#3b82f6" />
                    <Suspense fallback={null}>
                      {item.render3D(isHovered || isSelected)}
                    </Suspense>
                  </Canvas>

                  {/* Hover visual cue */}
                  <div
                    className={`absolute bottom-2 right-2 text-[10px] font-mono px-2 py-0.5 rounded bg-black/70 border border-cyan-500/30 text-cyan-300 transition-opacity ${
                      isHovered ? 'opacity-100' : 'opacity-0'
                    }`}
                  >
                    Interactive 3D
                  </div>
                </div>

                {/* Content details */}
                <div className="space-y-2 relative z-10 pt-2">
                  <h3 className="text-xl font-bold text-white font-heading flex items-center justify-between group-hover:text-cyan-300 transition-colors">
                    <span>{item.name}</span>
                    <ArrowRight className="w-4 h-4 text-slate-500 group-hover:text-cyan-400 group-hover:translate-x-1 transition-all" />
                  </h3>
                  <div className="text-xs font-medium text-cyan-400 font-mono">
                    {item.desc}
                  </div>
                  <p className="text-[11px] text-slate-400 leading-relaxed pt-1">
                    {item.details}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Optional Secondary Auto-Detect (Marked Coming Soon as strictly instructed in PART 5) */}
        <div className="pt-4 flex items-center justify-center">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-900/60 border border-white/10 text-slate-500 text-xs font-mono">
            <Clock className="w-3.5 h-3.5 text-slate-600" />
            <span>CROSS-MODALITY AUTO DETECT</span>
            <span className="text-[10px] px-1.5 py-0.2 rounded bg-white/5 border border-white/10 text-slate-400">
              Coming Soon
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
