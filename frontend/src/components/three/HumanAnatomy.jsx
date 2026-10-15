import React, { useRef, useState, useMemo, useEffect } from 'react';

import { useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import ScanBeam from './ScanBeam';
import FloatingParticles from './FloatingParticles';
import { HologramSkinShader } from './hologramShader';
import {
  createTorsoSurfaceGeometry,
  createLegSurfaceGeometry,
  createFootSurfaceGeometry,
  createArmSurfaceGeometry,
  createHandSurfaceGroup,
  createDetailedSkullGroup,
  createVertebralColumnGroup,
  createThoracicCageGroup,
  createPelvisGroup,
  createAppendicularSkeletonGroup,
  createMajorOrgansGroup,
  createVascularTreeGroup,
} from './anatomicalGeometry';

export default function HumanAnatomy({ isTransitioning = false, onHoverRegion }) {
  const groupRef = useRef();
  const lungsRef = useRef();
  const heartRef = useRef();
  const skinShaderRef = useRef();


  const [hoveredRegion, setHoveredRegion] = useState(null);
  const { pointer } = useThree();

  const handlePointerOver = (region, e) => {
    e.stopPropagation();
    setHoveredRegion(region);
    if (onHoverRegion) onHoverRegion(region);
  };

  const handlePointerOut = (e) => {
    e.stopPropagation();
    setHoveredRegion(null);
    if (onHoverRegion) onHoverRegion(null);
  };

  // Materials for realistic medical holographic rendering
  const materials = useMemo(() => {
    // 1. Holographic Skin Material with Fresnel Edge Rim
    const skinShaderMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uColor: { value: new THREE.Color('#00e5ff') },
        uDeepColor: { value: new THREE.Color('#021438') },
        uRimPower: { value: 2.7 },
        uOpacity: { value: 0.88 },
        uScanY: { value: 0.0 },
        uHoverBoost: { value: 0.0 },
      },
      vertexShader: HologramSkinShader.vertexShader,
      fragmentShader: HologramSkinShader.fragmentShader,
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });



    // 2. Bone Material: Glowing radiopaque bone with cyan-blue subsurface
    const createBoneMat = (isHighlighted = false) =>
      new THREE.MeshStandardMaterial({
        color: isHighlighted ? '#67e8f9' : '#e2e8f0',
        emissive: isHighlighted ? '#00f2ff' : '#0284c7',
        emissiveIntensity: isHighlighted ? 1.6 : 0.75,
        roughness: 0.25,
        metalness: 0.15,
        transparent: true,
        opacity: 0.85,
      });

    // 3. Spinal Discs Material
    const discMat = new THREE.MeshStandardMaterial({
      color: '#38bdf8',
      emissive: '#0369a1',
      emissiveIntensity: 0.8,
      transparent: true,
      opacity: 0.75,
    });

    // 4. Organ Materials with volumetric glow
    const organMats = {
      lung: new THREE.MeshStandardMaterial({
        color: '#06b6d4',
        emissive: '#0891b2',
        emissiveIntensity: 0.65,
        transparent: true,
        opacity: 0.55,
        roughness: 0.35,
      }),
      heart: new THREE.MeshStandardMaterial({
        color: '#818cf8',
        emissive: '#6366f1',
        emissiveIntensity: 0.95,
        transparent: true,
        opacity: 0.7,
      }),
      vessels: new THREE.MeshBasicMaterial({
        color: '#ef4444',
        transparent: true,
        opacity: 0.75,
      }),
      liver: new THREE.MeshStandardMaterial({
        color: '#0ea5e9',
        emissive: '#0284c7',
        emissiveIntensity: 0.5,
        transparent: true,
        opacity: 0.5,
      }),
      stomach: new THREE.MeshStandardMaterial({
        color: '#06b6d4',
        emissive: '#0e7490',
        emissiveIntensity: 0.45,
        transparent: true,
        opacity: 0.45,
      }),
      kidney: new THREE.MeshStandardMaterial({
        color: '#38bdf8',
        emissive: '#0284c7',
        emissiveIntensity: 0.6,
        transparent: true,
        opacity: 0.6,
      }),
      intestine: new THREE.MeshStandardMaterial({
        color: '#0284c7',
        emissive: '#0369a1',
        emissiveIntensity: 0.4,
        transparent: true,
        opacity: 0.45,
      }),
    };

    const vascularMat = new THREE.MeshBasicMaterial({
      color: '#00f2ff',
      transparent: true,
      opacity: 0.7,
    });

    return {
      skin: skinShaderMat,
      bone: createBoneMat,
      disc: discMat,
      organs: organMats,
      vascular: vascularMat,
    };
  }, []);

  useEffect(() => {
    skinShaderRef.current = materials.skin;
  }, [materials.skin]);

  // Geometries for continuous human surface & limbs

  const geometries = useMemo(() => {
    return {
      torso: createTorsoSurfaceGeometry(),
      leftLeg: createLegSurfaceGeometry(true),
      rightLeg: createLegSurfaceGeometry(false),
      leftFoot: createFootSurfaceGeometry(true),
      rightFoot: createFootSurfaceGeometry(false),
      leftArm: createArmSurfaceGeometry(true),
      rightArm: createArmSurfaceGeometry(false),
    };
  }, []);

  // Bone / Skeleton sub-groups
  const skeleton = useMemo(() => {
    const isSkullHov = hoveredRegion === 'skull';
    const isChestHov = hoveredRegion === 'chest';
    const isSpineHov = hoveredRegion === 'spine';
    const isPelvisHov = hoveredRegion === 'pelvis' || hoveredRegion === 'abdomen';

    return {
      skull: createDetailedSkullGroup(materials.bone(isSkullHov)),
      spine: createVertebralColumnGroup(materials.bone(isSpineHov), materials.disc),
      thoracic: createThoracicCageGroup(materials.bone(isChestHov)),
      pelvis: createPelvisGroup(materials.bone(isPelvisHov)),
      limbs: createAppendicularSkeletonGroup(materials.bone(false)),
      leftHand: createHandSurfaceGroup(true, materials.skin, materials.bone(false)),
      rightHand: createHandSurfaceGroup(false, materials.skin, materials.bone(false)),
      organs: createMajorOrgansGroup(materials.organs),
      vascular: createVascularTreeGroup(materials.vascular),
    };
  }, [hoveredRegion, materials]);

  // Frame update: breathing animation, cardiac pulse, scan beam tracking, cursor parallax
  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();

    // Update Hologram Skin Shader uniforms
    if (skinShaderRef.current?.uniforms) {
      skinShaderRef.current.uniforms.uTime.value = t;
      const scanY = Math.sin(t * 1.1) * 2.1;
      skinShaderRef.current.uniforms.uScanY.value = scanY;
      skinShaderRef.current.uniforms.uHoverBoost.value = hoveredRegion ? 1.0 : 0.0;
    }


    // Physiological Breathing Motion (Lungs expand softly)
    if (lungsRef.current) {
      const breath = 1.0 + Math.sin(t * 1.4) * 0.04;
      lungsRef.current.scale.set(breath, breath, breath);
    }

    // Cardiac Pulse (Heart beats subtly)
    if (heartRef.current) {
      const pulse = 1.0 + Math.sin(t * 4.5) * 0.06;
      heartRef.current.scale.set(pulse, pulse, pulse);
    }

    if (!groupRef.current) return;

    if (isTransitioning) {
      groupRef.current.position.z = THREE.MathUtils.lerp(groupRef.current.position.z, -2.0, 0.05);
      groupRef.current.position.y = THREE.MathUtils.lerp(groupRef.current.position.y, -0.3, 0.05);
      groupRef.current.scale.lerp(new THREE.Vector3(0.82, 0.82, 0.82), 0.05);
    } else {
      // Gentle idle sway + subtle cursor parallax
      const targetRotY = Math.sin(t * 0.35) * 0.12 + pointer.x * 0.22;
      const targetRotX = pointer.y * -0.10;

      groupRef.current.rotation.y = THREE.MathUtils.lerp(groupRef.current.rotation.y, targetRotY, 0.04);
      groupRef.current.rotation.x = THREE.MathUtils.lerp(groupRef.current.rotation.x, targetRotX, 0.04);
    }
  });

  return (
    <>
      <OrbitControls
        enablePan={false}
        enableZoom={true}
        minDistance={3.0}
        maxDistance={7.5}
        minPolarAngle={Math.PI / 4}
        maxPolarAngle={(3 * Math.PI) / 4}
        dampingFactor={0.06}
        enableDamping={true}
        rotateSpeed={0.8}
      />

      <group ref={groupRef} position={[0, -0.05, 0]}>
        {/* Periodic Medical Scan Beam */}
        <ScanBeam minY={-2.15} maxY={2.05} speed={1.1} />
        <FloatingParticles count={60} />

        {/* ======================================================== */}
        {/* 1. CONTINUOUS SEMI-TRANSPARENT HOLOGRAPHIC SKIN SURFACE   */}
        {/* ======================================================== */}
        <group>
          {/* Continuous Torso, Neck & Head Surface */}
          <mesh geometry={geometries.torso} material={materials.skin} />

          {/* Bilateral Legs (Thighs, Knees, Calves, Ankles) */}
          <mesh geometry={geometries.leftLeg} material={materials.skin} />
          <mesh geometry={geometries.rightLeg} material={materials.skin} />

          {/* Bilateral Feet (Heel, Arch, Metatarsus, Toes) */}
          <mesh geometry={geometries.leftFoot} material={materials.skin} />
          <mesh geometry={geometries.rightFoot} material={materials.skin} />

          {/* Bilateral Arms (Deltoid, Upper Arm, Forearm, Wrist) */}
          <mesh geometry={geometries.leftArm} material={materials.skin} />
          <mesh geometry={geometries.rightArm} material={materials.skin} />

          {/* Bilateral Hands with 5 distinct articulated fingers */}
          <primitive object={skeleton.leftHand} />
          <primitive object={skeleton.rightHand} />
        </group>

        {/* ======================================================== */}
        {/* 2. REALISTIC GLOWING SKELETAL SYSTEM                     */}
        {/* ======================================================== */}
        {/* Skull & Facial Skeleton (Hoverable) */}
        <group
          onPointerOver={(e) => handlePointerOver('skull', e)}
          onPointerOut={handlePointerOut}
        >
          <primitive object={skeleton.skull} />
        </group>

        {/* Vertebral Column / Spine (Hoverable) */}
        <group
          onPointerOver={(e) => handlePointerOver('spine', e)}
          onPointerOut={handlePointerOut}
        >
          <primitive object={skeleton.spine} />
        </group>

        {/* Thoracic Cage: Ribs, Sternum, Clavicles, Scapulae (Hoverable) */}
        <group
          onPointerOver={(e) => handlePointerOver('chest', e)}
          onPointerOut={handlePointerOut}
        >
          <primitive object={skeleton.thoracic} />
        </group>

        {/* Pelvic Girdle */}
        <primitive object={skeleton.pelvis} />

        {/* Appendicular Skeleton: Femurs, Tibias, Fibulas, Humeri, Radii, Ulnae */}
        <primitive object={skeleton.limbs} />

        {/* ======================================================== */}
        {/* 3. MAJOR INTERNAL ORGANS                                 */}
        {/* ======================================================== */}
        <group
          onPointerOver={(e) => handlePointerOver('abdomen', e)}
          onPointerOut={handlePointerOut}
        >
          <primitive object={skeleton.organs} />
        </group>

        {/* ======================================================== */}
        {/* 4. CARDIOVASCULAR VASCULAR TREE                          */}
        {/* ======================================================== */}
        <primitive object={skeleton.vascular} />
      </group>
    </>
  );
}
