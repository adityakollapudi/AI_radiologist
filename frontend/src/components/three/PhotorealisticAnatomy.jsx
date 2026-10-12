import React, { useRef, useState, useEffect, useCallback } from 'react';
import { ChevronRight } from 'lucide-react';
import frameNames from './frameList.json';

export default function PhotorealisticAnatomy({
  isTransitioning = false,
  onHoverRegion,
  onSelectModality,
  activeRegion: _activeRegion,
}) {
  const containerRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const loadedFramesRef = useRef([]);
  const resumeTimerRef = useRef(null);

  const [isPlaying, setIsPlaying] = useState(true);
  const [rotationAngle, setRotationAngle] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [startAngle, setStartAngle] = useState(0);
  const [renderMode, setRenderMode] = useState('video'); // 'video' | 'frames'
  const [scanBeamY, setScanBeamY] = useState(20);
  const [mediaAspect, setMediaAspect] = useState(null);

  const totalFrames = frameNames.length; // 75 high-res frames

  // Preload frames in background for instantaneous zero-latency drag-scrubbing
  useEffect(() => {
    const cache = [];
    frameNames.forEach((filename, idx) => {
      const img = new Image();
      img.src = `/anatomy_frames/${filename}`;
      cache[idx] = img;
    });
    loadedFramesRef.current = cache;
  }, []);

  // Continuous scan beam animation
  useEffect(() => {
    let animId;
    let start = performance.now();

    const animateBeam = (now) => {
      const elapsed = (now - start) / 1000;
      // Oscillate beam between 8% and 92% of height
      const y = 50 + Math.sin(elapsed * 1.4) * 42;
      setScanBeamY(y);
      animId = requestAnimationFrame(animateBeam);
    };

    animId = requestAnimationFrame(animateBeam);
    return () => cancelAnimationFrame(animId);
  }, []);

  // Update canvas when in 'frames' mode or when dragging
  const drawFrame = useCallback((angle) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Normalize angle to 0..360
    const normalized = ((angle % 360) + 360) % 360;
    const frameIndex = Math.min(
      totalFrames - 1,
      Math.floor((normalized / 360) * totalFrames)
    );

    const img = loadedFramesRef.current[frameIndex];
    if (img && img.complete && img.naturalWidth > 0) {
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
    }
  }, [totalFrames]);

  // Sync angle changes to canvas & video
  useEffect(() => {
    if (renderMode === 'frames') {
      drawFrame(rotationAngle);
    } else if (videoRef.current && !isPlaying) {
      const video = videoRef.current;
      if (video.duration) {
        const targetTime = (((rotationAngle % 360) + 360) % 360 / 360) * video.duration;
        video.currentTime = targetTime;
      }
    }
  }, [rotationAngle, renderMode, isPlaying, drawFrame]);

  // Video timeupdate tracking to keep angle in sync during continuous autoplay
  const handleVideoTimeUpdate = () => {
    if (isPlaying && videoRef.current && videoRef.current.duration) {
      const fraction = videoRef.current.currentTime / videoRef.current.duration;
      const deg = Math.round(fraction * 360);
      setRotationAngle(deg);
    }
  };

  const handleMediaLoad = (e) => {
    const width = e.target.videoWidth || e.target.naturalWidth;
    const height = e.target.videoHeight || e.target.naturalHeight;
    if (width && height) {
      setMediaAspect(width / height);
    }
  };

  // Pointer / Mouse Drag scrub handlers
  const handlePointerDown = (e) => {
    setIsDragging(true);
    setDragStartX(e.clientX || (e.touches && e.touches[0].clientX) || 0);
    setStartAngle(rotationAngle);

    if (resumeTimerRef.current) {
      clearTimeout(resumeTimerRef.current);
      resumeTimerRef.current = null;
    }

    if (isPlaying) {
      setIsPlaying(false);
      if (videoRef.current) {
        videoRef.current.pause();
      }
    }
    // Switch to frame scrubber during active drag for frame-accurate responsiveness
    setRenderMode('frames');
    drawFrame(rotationAngle);
  };

  const handlePointerMove = (e) => {
    if (!isDragging) return;
    const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
    const deltaX = clientX - dragStartX;

    // 1px delta = 0.7 degrees of horizontal anatomical rotation
    const newAngle = ((startAngle + deltaX * 0.7) % 360 + 360) % 360;
    setRotationAngle(newAngle);
    drawFrame(newAngle);
  };

  const handlePointerUp = () => {
    if (!isDragging) return;
    setIsDragging(false);

    // After 1.5 seconds of inactivity, smoothly resume video auto-rotation
    if (resumeTimerRef.current) clearTimeout(resumeTimerRef.current);
    resumeTimerRef.current = setTimeout(() => {
      if (videoRef.current && videoRef.current.duration) {
        const targetTime = (((rotationAngle % 360) + 360) % 360 / 360) * videoRef.current.duration;
        videoRef.current.currentTime = targetTime;
        videoRef.current.play().catch(() => {});
      }
      setRenderMode('video');
      setIsPlaying(true);
    }, 1500);
  };

  // Anatomical hotspots coordinates (relative percentages from top)
  const hotspots = [
    {
      id: 'skull',
      label: 'Cranial Cavity',
      target: 'Brain MRI Target',
      modality: 'mri',
      top: '13%',
      left: '50%',
    },
    {
      id: 'chest',
      label: 'Thoracic Cavity',
      target: 'Chest X-Ray / CT Target',
      modality: 'ct',
      top: '28%',
      left: '50%',
    },
    {
      id: 'spine',
      label: 'Vertebral Column',
      target: 'MURA Skeletal Target',
      modality: 'xray',
      top: '44%',
      left: '50%',
    },
    {
      id: 'abdomen',
      label: 'Abdominal Viscera',
      target: 'Deep Axial Slice',
      modality: 'ct',
      top: '56%',
      left: '50%',
    },
  ];

  const handleHotspotEnter = (spot) => {
    if (onHoverRegion) onHoverRegion(spot.id);
  };

  const handleHotspotLeave = () => {
    if (onHoverRegion) onHoverRegion(null);
  };

  const handleHotspotClick = (modality) => {
    if (onSelectModality) {
      onSelectModality(modality);
    }
  };

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
      className={`relative w-full h-full select-none flex items-center justify-center overflow-hidden rounded-2xl bg-[#030712]/90 border border-cyan-500/20 shadow-[0_0_50px_rgba(6,182,212,0.15)] transition-all duration-500 ${
        isTransitioning ? 'scale-90 opacity-40 blur-xs' : 'scale-100 opacity-100'
      } ${isDragging ? 'cursor-grabbing' : 'cursor-grab'}`}
    >
      {/* Radial Holographic Core Glow */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(6,182,212,0.18)_0%,rgba(14,165,233,0.06)_45%,transparent_75%)] pointer-events-none" />

      {/* Cyber Grid Texture */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#08334415_1px,transparent_1px),linear-gradient(to_bottom,#08334415_1px,transparent_1px)] bg-[size:28px_28px] pointer-events-none" />

      {/* Black patch to cover Spline watermark - sits above video but below scan beam */}
      <div
        className="absolute z-[15] pointer-events-none"
        style={{ bottom: '10%', right: '23%', width: '40px', height: '40px', backgroundColor: '#030712' }}
      />

      {/* Sweeping Laser Scan Beam - z-20 so it passes in front of the black patch */}
      <div
        className="absolute left-0 right-0 z-20 pointer-events-none transition-transform duration-75"
        style={{ top: `${scanBeamY}%` }}
      >
        <div className="h-[2px] w-full bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_15px_#00e5ff]" />
        <div className="h-[24px] w-full bg-gradient-to-b from-cyan-400/15 to-transparent pointer-events-none" />
      </div>

      {/* Main Rotating Translucent Body Viewport */}
      <div className="relative w-full h-full flex items-center justify-center p-2">
        {/* Hardware-accelerated 1080p Video Mode */}
        <video
          ref={videoRef}
          src="/anatomy_rotation.mp4"
          autoPlay
          loop
          muted
          playsInline
          onTimeUpdate={handleVideoTimeUpdate}
          className={`max-h-[95%] max-w-[95%] object-contain mix-blend-screen filter drop-shadow-[0_0_35px_rgba(6,182,212,0.4)] transition-opacity duration-300 ${
            renderMode === 'video' ? 'opacity-100 block' : 'opacity-0 hidden'
          }`}
        />

        {/* 360° Frame Scrubber Canvas Mode (Active during drag & scrub) */}
        <canvas
          ref={canvasRef}
          className={`max-h-[95%] max-w-[95%] object-contain mix-blend-screen filter drop-shadow-[0_0_35px_rgba(6,182,212,0.4)] transition-opacity duration-300 ${
            renderMode === 'frames' ? 'opacity-100 block' : 'opacity-0 hidden'
          }`}
        />

        {/* Interactive Hotspot Targets */}
        <div className="absolute inset-0 pointer-events-none">
          {hotspots.map((spot) => (
            <div
              key={spot.id}
              style={{ top: spot.top, left: spot.left }}
              className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-auto"
            >
              <div
                onMouseEnter={() => handleHotspotEnter(spot)}
                onMouseLeave={handleHotspotLeave}
                onClick={() => handleHotspotClick(spot.modality)}
                className="group relative flex items-center justify-center cursor-pointer p-2"
              >
                {/* Pulsing Target Bullseye */}
                <div className="w-4 h-4 rounded-full border border-cyan-400/50 bg-cyan-500/15 group-hover:scale-125 group-hover:border-cyan-300 group-hover:bg-cyan-400/35 transition-all duration-300 flex items-center justify-center shadow-[0_0_10px_rgba(6,182,212,0.4)]">
                  <div className="w-1 h-1 rounded-full bg-cyan-300 animate-ping" />
                </div>

                {/* Target Label Callout */}
                <div className="absolute left-6 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 translate-x-1 group-hover:translate-x-0 transition-all duration-200 pointer-events-none z-40 whitespace-nowrap">
                  <div className="px-2.5 py-1 rounded-lg bg-slate-950/95 border border-cyan-400/50 text-cyan-200 text-xs font-mono shadow-[0_4px_20px_rgba(0,0,0,0.8)] backdrop-blur-md space-y-0.5">
                    <div className="font-bold text-cyan-300 uppercase flex items-center gap-1">
                      <span>{spot.label}</span>
                      <ChevronRight className="w-3 h-3 text-cyan-400" />
                    </div>
                    <div className="text-[10px] text-slate-300">{spot.target}</div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
