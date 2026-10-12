import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

export default function CTScene({ isHovered = false }) {
  const gantryRef = useRef();
  const slicesRef = useRef();

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();

    if (gantryRef.current) {
      // Rotating CT Gantry Ring
      const speed = isHovered ? 1.6 : 0.6;
      gantryRef.current.rotation.z = t * speed;
    }

    if (slicesRef.current) {
      // Gentle axial floating / slice separation
      slicesRef.current.rotation.y = Math.sin(t * 0.5) * 0.25;
      const spread = isHovered ? 0.35 : 0.24;
      slicesRef.current.children.forEach((child, i) => {
        child.position.z = (i - 2) * spread;
      });
    }
  });

  return (
    <group>
      {/* Outer CT Scanner Gantry Ring */}
      <group ref={gantryRef}>
        <mesh>
          <torusGeometry args={[1.35, 0.12, 16, 48]} />
          <meshStandardMaterial
            color="#0f172a"
            emissive="#0284c7"
            emissiveIntensity={isHovered ? 1.2 : 0.6}
            roughness={0.2}
            metalness={0.8}
          />
        </mesh>
        {/* Glowing detector markers on the ring */}
        {[0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2].map((angle, i) => (
          <mesh
            key={`marker-${i}`}
            position={[Math.cos(angle) * 1.35, Math.sin(angle) * 1.35, 0]}
          >
            <sphereGeometry args={[0.07, 10, 10]} />
            <meshBasicMaterial color="#00f2ff" />
          </mesh>
        ))}
      </group>

      {/* Layered Axial 2D CT Slices through the Bore */}
      <group ref={slicesRef} rotation={[-0.3, 0.4, 0]}>
        {[-2, -1, 0, 1, 2].map((idx) => {
          const isCenter = idx === 0;
          return (
            <group key={`slice-${idx}`} position={[0, 0, idx * 0.24]}>
              {/* Slice Ring Border */}
              <mesh>
                <ringGeometry args={[0.78, 0.82, 32]} />
                <meshBasicMaterial
                  color={isCenter ? '#38bdf8' : '#0284c7'}
                  transparent
                  opacity={isHovered ? 0.9 : 0.5}
                  side={THREE.DoubleSide}
                />
              </mesh>
              {/* Internal Cross-section Slice Plane with Axial anatomy */}
              <mesh>
                <circleGeometry args={[0.77, 32]} />
                <meshStandardMaterial
                  color="#091428"
                  emissive={isCenter ? '#06b6d4' : '#0369a1'}
                  emissiveIntensity={isCenter ? 0.6 : 0.2}
                  transparent
                  opacity={0.65}
                  side={THREE.DoubleSide}
                />
              </mesh>
              {/* Internal Axial Structure (Spine + Bilateral lung fields) */}
              <mesh position={[0, -0.32, 0.01]}>
                <sphereGeometry args={[0.1, 10, 10]} />
                <meshBasicMaterial color="#e0f2fe" />
              </mesh>
              <mesh position={[-0.24, 0.08, 0.01]}>
                <circleGeometry args={[0.22, 16]} />
                <meshBasicMaterial color="#020617" />
              </mesh>
              <mesh position={[0.24, 0.08, 0.01]}>
                <circleGeometry args={[0.22, 16]} />
                <meshBasicMaterial color="#020617" />
              </mesh>
            </group>
          );
        })}
      </group>
    </group>
  );
}
