import React, { useRef, useState, useEffect, useCallback } from 'react';

const TOTAL_FRAMES = 100;
const frameNames = Array.from({ length: TOTAL_FRAMES }, (_, i) => `frame_${String(i).padStart(3, '0')}.webp`);

export default function Brain3DViewer({ isHovered = false, isSelected = false }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const loadedFramesRef = useRef([]);
  const [framesLoaded, setFramesLoaded] = useState(false);
  const [rotationAngle, setRotationAngle] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [startAngle, setStartAngle] = useState(0);
  const [scanBeamY, setScanBeamY] = useState(25);
  const animFrameIdRef = useRef(null);
  const lastTimeRef = useRef(performance.now());
  const angleRef = useRef(0);

  const totalFrames = TOTAL_FRAMES;

  // Preload all 100 frames
  useEffect(() => {
    let loadedCount = 0;
    const cache = [];
    frameNames.forEach((filename, idx) => {
      const img = new Image();
      img.src = `/brain_frames/${filename}`;
      img.onload = () => {
        loadedCount += 1;
        if (loadedCount >= Math.min(20, totalFrames)) {
          setFramesLoaded(true);
        }
      };
      cache[idx] = img;
    });
    loadedFramesRef.current = cache;
  }, [totalFrames]);

  // Keep angleRef synced
  useEffect(() => {
    angleRef.current = rotationAngle;
  }, [rotationAngle]);

  // Draw frame to canvas
  const drawFrame = useCallback((deg) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const normalized = ((deg % 360) + 360) % 360;
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

  // Continuous auto-rotation animation loop when not dragging
  useEffect(() => {
    if (isDragging) return;

    lastTimeRef.current = performance.now();
    const speed = isHovered ? 45 : 28; // deg per sec

    const loop = (now) => {
      const dt = (now - lastTimeRef.current) / 1000;
      lastTimeRef.current = now;

      const newAngle = (angleRef.current + dt * speed) % 360;
      angleRef.current = newAngle;
      setRotationAngle(newAngle);
      drawFrame(newAngle);

      animFrameIdRef.current = requestAnimationFrame(loop);
    };

    animFrameIdRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animFrameIdRef.current);
  }, [isDragging, isHovered, drawFrame]);

  // Sweeping Laser Scan Line (MRI slice acquisition effect)
  useEffect(() => {
    let animId;
    let start = performance.now();

    const animateBeam = (now) => {
      const elapsed = (now - start) / 1000;
      const speed = isHovered ? 2.2 : 1.4;
      const y = 50 + Math.sin(elapsed * speed) * 38;
      setScanBeamY(y);
      animId = requestAnimationFrame(animateBeam);
    };

    animId = requestAnimationFrame(animateBeam);
    return () => cancelAnimationFrame(animId);
  }, [isHovered]);

  // Pointer drag scrubbing
  const handlePointerDown = (e) => {
    e.stopPropagation();
    setIsDragging(true);
    const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
    setDragStartX(clientX);
    setStartAngle(angleRef.current);
  };

  const handlePointerMove = (e) => {
    if (!isDragging) return;
    const clientX = e.clientX || (e.touches && e.touches[0].clientX) || 0;
    const deltaX = clientX - dragStartX;

    // 1px delta = 0.8 degrees of rotation
    const newAngle = ((startAngle + deltaX * 0.8) % 360 + 360) % 360;
    angleRef.current = newAngle;
    setRotationAngle(newAngle);
    drawFrame(newAngle);
  };

  const handlePointerUp = () => {
    if (!isDragging) return;
    setIsDragging(false);
    lastTimeRef.current = performance.now();
  };

  return (
    <div
      ref={containerRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
      className={`relative w-full h-full select-none flex items-center justify-center overflow-hidden rounded-xl bg-slate-950 transition-all duration-300 ${
        isDragging ? 'cursor-grabbing' : 'cursor-grab'
      }`}
    >
      {/* Background Holographic Atmosphere */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(6,182,212,0.18)_0%,rgba(3,7,18,0.95)_75%)] pointer-events-none" />

      {/* Cyber Grid Pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#08334415_1px,transparent_1px),linear-gradient(to_bottom,#08334415_1px,transparent_1px)] bg-[size:24px_24px] pointer-events-none" />

      {/* Scanning Laser Beam */}
      <div
        className="absolute left-0 right-0 z-20 pointer-events-none transition-transform duration-75"
        style={{ top: `${scanBeamY}%` }}
      >
        <div className="h-[2px] w-full bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_12px_#00e5ff]" />
        <div className="h-[18px] w-full bg-gradient-to-b from-cyan-400/20 to-transparent pointer-events-none" />
      </div>

      {/* 3D Brain Visual Canvas / Animated Fallback */}
      <div className="relative w-full h-full flex items-center justify-center p-1 pointer-events-none">
        {/* Hardware-accelerated canvas for 360 degree rotation and scrub */}
        <canvas
          ref={canvasRef}
          className={`max-h-[92%] max-w-[92%] object-contain filter drop-shadow-[0_0_25px_rgba(6,182,212,0.5)] transition-opacity duration-300 ${
            framesLoaded ? 'opacity-100 block' : 'opacity-0 hidden'
          }`}
        />

        {/* Fallback Animated WebP while canvas preloads */}
        {!framesLoaded && (
          <img
            src="/brain_rotation.webp"
            alt="3D Brain MRI Volumetric Projection"
            className="max-h-[92%] max-w-[92%] object-contain filter drop-shadow-[0_0_25px_rgba(6,182,212,0.5)]"
          />
        )}
      </div>
    </div>
  );
}
