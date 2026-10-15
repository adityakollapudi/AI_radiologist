import React, { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// Deterministic pseudorandom generator for React purity
function pseudoRandom(seed) {
  const x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}

export default function FloatingParticles({ count = 65, bounds = [3.5, 4.5, 3.5] }) {
  const pointsRef = useRef();

  const [positions, phases] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const ph = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (pseudoRandom(i * 3 + 1) - 0.5) * bounds[0];
      pos[i * 3 + 1] = (pseudoRandom(i * 3 + 2) - 0.5) * bounds[1];
      pos[i * 3 + 2] = (pseudoRandom(i * 3 + 3) - 0.5) * bounds[2];
      ph[i] = pseudoRandom(i * 7 + 4) * Math.PI * 2;
    }
    return [pos, ph];
  }, [count, bounds]);

  useFrame(({ clock }) => {
    if (!pointsRef.current) return;
    const t = clock.getElapsedTime() * 0.4;
    const geo = pointsRef.current.geometry;
    const posArray = geo.attributes.position.array;

    for (let i = 0; i < count; i++) {
      const idx = i * 3 + 1; // Y axis
      posArray[idx] += Math.sin(t + phases[i]) * 0.002;
    }
    geo.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.035}
        color="#38bdf8"
        transparent
        opacity={0.55}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}
