import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export default function XRayScene({ isHovered = false }) {
  const groupRef = useRef();
  const scanLineRef = useRef();

  useFrame(({ clock }) => {
    if (!groupRef.current) return;
    const t = clock.getElapsedTime();

    // Idle rotation + hover reactivity
    const rotSpeed = isHovered ? 1.0 : 0.4;
    groupRef.current.rotation.y = Math.sin(t * rotSpeed) * 0.22;
    groupRef.current.rotation.x = isHovered ? -0.1 : 0;

    // Scan line animation
    if (scanLineRef.current) {
      scanLineRef.current.position.y = Math.sin(t * 2) * 0.9;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Background Radiographic Detector Cassette Plate */}
      <mesh position={[0, 0, -0.45]}>
        <boxGeometry args={[2.2, 2.5, 0.05]} />
        <meshStandardMaterial
          color="#030b18"
          roughness={0.7}
          metalness={0.8}
        />
      </mesh>
      {/* Detector Grid Frame */}
      <lineSegments position={[0, 0, -0.42]}>
        <edgesGeometry args={[new THREE.BoxGeometry(2.2, 2.5, 0.05)]} />
        <lineBasicMaterial color={isHovered ? '#00f2ff' : '#0284c7'} />
      </lineSegments>

      {/* Moving X-Ray Laser Line */}
      <mesh ref={scanLineRef} position={[0, 0, -0.38]}>
        <planeGeometry args={[2.1, 0.04]} />
        <meshBasicMaterial
          color="#38bdf8"
          transparent
          opacity={isHovered ? 0.95 : 0.65}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Anatomical Thorax: Spine Midline */}
      <mesh position={[0, 0, -0.1]}>
        <cylinderGeometry args={[0.06, 0.06, 1.8, 12]} />
        <meshStandardMaterial
          color="#e0f2fe"
          emissive="#0284c7"
          emissiveIntensity={isHovered ? 1.4 : 0.8}
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* Bilateral Ribs */}
      {Array.from({ length: 7 }).map((_, i) => {
        const y = 0.7 - i * 0.22;
        const w = 0.65 - i * 0.02;
        return (
          <group key={`xrib-${i}`} position={[0, y, 0]}>
            {/* Left Rib */}
            <mesh position={[-0.45, 0, 0.05]} rotation={[0.08, 0, 0.1]}>
              <torusGeometry args={[w, 0.028, 8, 16, Math.PI * 0.85]} />
              <meshStandardMaterial
                color="#e0f2fe"
                emissive="#06b6d4"
                emissiveIntensity={isHovered ? 1.5 : 0.7}
                transparent
                opacity={0.9}
              />
            </mesh>
            {/* Right Rib */}
            <mesh position={[0.45, 0, 0.05]} rotation={[0.08, 0, -0.1]}>
              <torusGeometry args={[w, 0.028, 8, 16, Math.PI * 0.85]} />
              <meshStandardMaterial
                color="#e0f2fe"
                emissive="#06b6d4"
                emissiveIntensity={isHovered ? 1.5 : 0.7}
                transparent
                opacity={0.9}
              />
            </mesh>
          </group>
        );
      })}

      {/* Lungs Radiographic Silhouette */}
      <mesh position={[-0.38, 0.05, 0]}>
        <sphereGeometry args={[0.38, 14, 14]} />
        <meshStandardMaterial
          color="#0891b2"
          emissive="#06b6d4"
          emissiveIntensity={isHovered ? 0.7 : 0.3}
          transparent
          opacity={0.35}
        />
      </mesh>
      <mesh position={[0.38, 0.05, 0]}>
        <sphereGeometry args={[0.38, 14, 14]} />
        <meshStandardMaterial
          color="#0891b2"
          emissive="#06b6d4"
          emissiveIntensity={isHovered ? 0.7 : 0.3}
          transparent
          opacity={0.35}
        />
      </mesh>
    </group>
  );
}
