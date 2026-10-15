import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * Procedural horizontal medical scanning beam that sweeps up and down
 */
export default function ScanBeam({ minY = -2.1, maxY = 2.1, speed = 1.2 }) {
  const beamRef = useRef();
  const ringRef = useRef();

  useFrame(({ clock }) => {
    if (!beamRef.current) return;
    const t = clock.getElapsedTime() * speed;
    // Oscillate between minY and maxY smoothly
    const y = (Math.sin(t) * 0.5 + 0.5) * (maxY - minY) + minY;
    beamRef.current.position.y = y;

    if (ringRef.current) {
      ringRef.current.position.y = y;
      const pulse = 1 + Math.sin(t * 4) * 0.04;
      ringRef.current.scale.set(pulse, pulse, pulse);
    }
  });

  return (
    <group>
      {/* Planar Laser Sheet */}
      <mesh ref={beamRef} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[2.4, 1.4]} />
        <meshBasicMaterial
          color="#00f3ff"
          transparent
          opacity={0.35}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* Outer Scan Line Ring */}
      <mesh ref={ringRef} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.7, 0.76, 64]} />
        <meshBasicMaterial
          color="#06b6d4"
          transparent
          opacity={0.65}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}
