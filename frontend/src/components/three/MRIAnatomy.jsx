import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';


export default function MRIAnatomy({ isHovered = false }) {
  const brainRef = useRef();
  const coilRef = useRef();

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();

    if (brainRef.current) {
      const rotSpeed = isHovered ? 0.9 : 0.4;
      brainRef.current.rotation.y = Math.sin(t * rotSpeed) * 0.35;
      brainRef.current.rotation.x = Math.cos(t * 0.3) * 0.1;
    }

    if (coilRef.current) {
      coilRef.current.rotation.z = t * 0.5;
      coilRef.current.rotation.x = Math.PI / 2 + Math.sin(t) * 0.1;
    }
  });

  return (
    <group>
      {/* Magnetic Resonance RF Coil Rings */}
      <group ref={coilRef}>
        <mesh>
          <torusGeometry args={[1.3, 0.03, 12, 36]} />
          <meshBasicMaterial
            color="#818cf8"
            transparent
            opacity={isHovered ? 0.85 : 0.45}
          />
        </mesh>
        <mesh rotation={[0.4, 0.4, 0]}>
          <torusGeometry args={[1.35, 0.02, 12, 36]} />
          <meshBasicMaterial
            color="#06b6d4"
            transparent
            opacity={isHovered ? 0.75 : 0.35}
          />
        </mesh>
      </group>

      {/* Holographic Brain Structure */}
      <group ref={brainRef} scale={1.15}>
        {/* Left Cerebral Hemisphere */}
        <mesh position={[-0.32, 0.12, 0]}>
          <sphereGeometry args={[0.55, 20, 20]} />
          <meshStandardMaterial
            color="#0891b2"
            emissive={isHovered ? '#22d3ee' : '#0284c7'}
            emissiveIntensity={isHovered ? 1.3 : 0.75}
            roughness={0.4}
            metalness={0.1}
            wireframe={true}
            transparent
            opacity={0.8}
          />
        </mesh>
        {/* Left Hemisphere Inner Solid Glow */}
        <mesh position={[-0.32, 0.12, 0]}>
          <sphereGeometry args={[0.51, 16, 16]} />
          <meshStandardMaterial
            color="#0e7490"
            emissive="#06b6d4"
            emissiveIntensity={0.4}
            transparent
            opacity={0.35}
          />
        </mesh>

        {/* Right Cerebral Hemisphere */}
        <mesh position={[0.32, 0.12, 0]}>
          <sphereGeometry args={[0.55, 20, 20]} />
          <meshStandardMaterial
            color="#0891b2"
            emissive={isHovered ? '#22d3ee' : '#0284c7'}
            emissiveIntensity={isHovered ? 1.3 : 0.75}
            roughness={0.4}
            metalness={0.1}
            wireframe={true}
            transparent
            opacity={0.8}
          />
        </mesh>
        {/* Right Hemisphere Inner Solid Glow */}
        <mesh position={[0.32, 0.12, 0]}>
          <sphereGeometry args={[0.51, 16, 16]} />
          <meshStandardMaterial
            color="#0e7490"
            emissive="#06b6d4"
            emissiveIntensity={0.4}
            transparent
            opacity={0.35}
          />
        </mesh>

        {/* Cerebellum (Posterior-Inferior) */}
        <mesh position={[0, -0.42, -0.22]}>
          <sphereGeometry args={[0.34, 14, 14]} />
          <meshStandardMaterial
            color="#1e293b"
            emissive="#0284c7"
            emissiveIntensity={isHovered ? 1.1 : 0.5}
            wireframe={true}
            transparent
            opacity={0.7}
          />
        </mesh>

        {/* Brainstem (Inferior Midline) */}
        <mesh position={[0, -0.68, 0]}>
          <cylinderGeometry args={[0.12, 0.1, 0.42, 14]} />
          <meshStandardMaterial
            color="#38bdf8"
            emissive="#0284c7"
            emissiveIntensity={0.6}
            transparent
            opacity={0.8}
          />
        </mesh>
      </group>
    </group>
  );
}
