import * as THREE from 'three';

/**
 * Creates a parametric lofted geometry through a series of cross-sectional rings.
 * Each ring has a center [x, y, z], radiusX, radiusZ, and rotation.
 */
export function createLoftedRingGeometry(rings, segments = 32) {
  const vertices = [];
  const indices = [];
  const uvs = [];

  const numRings = rings.length;

  for (let i = 0; i < numRings; i++) {
    const ring = rings[i];
    const { x = 0, y = 0, z = 0, rx = 0.2, rz = 0.2, rotZ = 0, offsetX = 0, offsetZ = 0 } = ring;

    for (let j = 0; j <= segments; j++) {
      const angle = (j / segments) * Math.PI * 2;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      // Local ring coordinates
      let lx = cosA * rx + offsetX;
      let lz = sinA * rz + offsetZ;
      let ly = 0;

      // Apply rotation if needed
      if (rotZ !== 0) {
        const tempX = lx * Math.cos(rotZ) - ly * Math.sin(rotZ);
        ly = lx * Math.sin(rotZ) + ly * Math.cos(rotZ);
        lx = tempX;
      }

      vertices.push(x + lx, y + ly, z + lz);
      uvs.push(j / segments, i / (numRings - 1));
    }
  }

  // Create triangles
  for (let i = 0; i < numRings - 1; i++) {
    for (let j = 0; j < segments; j++) {
      const a = i * (segments + 1) + j;
      const b = (i + 1) * (segments + 1) + j;
      const c = (i + 1) * (segments + 1) + (j + 1);
      const d = i * (segments + 1) + (j + 1);

      indices.push(a, b, d);
      indices.push(b, c, d);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  return geometry;
}

/**
 * Realistic Human Torso, Neck & Head Surface Geometry
 * Anatomic proportions matching the reference diagrams (pectoralis, waist, iliac crests, shoulders, neck, head)
 */
export function createTorsoSurfaceGeometry() {
  const rings = [
    // Groin / Perineum
    { y: -0.22, rx: 0.28, rz: 0.20, z: 0.02 },
    // Lower Pelvis / Pubic area
    { y: -0.10, rx: 0.33, rz: 0.22, z: 0.02 },
    // Iliac Crests / Hips
    { y: 0.08, rx: 0.35, rz: 0.23, z: 0.01 },
    // Waist narrowing (natural anatomical waist)
    { y: 0.26, rx: 0.31, rz: 0.21, z: 0.0 },
    // Lower Rib Cage margin (costal arch)
    { y: 0.44, rx: 0.34, rz: 0.23, z: 0.02 },
    // Mid Thorax
    { y: 0.65, rx: 0.38, rz: 0.25, z: 0.03 },
    // Chest / Pectoralis major apex
    { y: 0.88, rx: 0.42, rz: 0.27, z: 0.04 },
    // Clavicular level & Upper chest
    { y: 1.08, rx: 0.40, rz: 0.23, z: 0.02 },
    // Shoulders base / Trapezius slope
    { y: 1.22, rx: 0.32, rz: 0.18, z: -0.01 },
    // Lower Neck
    { y: 1.30, rx: 0.16, rz: 0.15, z: -0.01 },
    // Mid Neck / Thyroid cartilage (Adam's apple)
    { y: 1.38, rx: 0.13, rz: 0.14, z: 0.0 },
    // Submental / Jawline base
    { y: 1.48, rx: 0.16, rz: 0.17, z: 0.02 },
    // Facial level (chin, mouth, nose bridge)
    { y: 1.58, rx: 0.19, rz: 0.22, z: 0.04 },
    // Eye orbitals / Brow ridge
    { y: 1.68, rx: 0.22, rz: 0.25, z: 0.02 },
    // Forehead / Parietal cranial vault
    { y: 1.78, rx: 0.23, rz: 0.26, z: 0.0 },
    // Crown of head
    { y: 1.88, rx: 0.17, rz: 0.20, z: -0.01 },
    // Top apex
    { y: 1.94, rx: 0.05, rz: 0.06, z: -0.01 },
  ];

  return createLoftedRingGeometry(rings, 36);
}

/**
 * Anatomical Limb Surface: Leg (Thigh, Knee, Calf, Ankle)
 */
export function createLegSurfaceGeometry(isLeft = true) {
  const sign = isLeft ? -1 : 1;
  const rings = [
    // Upper thigh / Hip attachment
    { y: -0.18, rx: 0.17, rz: 0.17, x: sign * 0.19, z: 0.02 },
    // Mid Thigh (quadriceps muscular belly)
    { y: -0.42, rx: 0.15, rz: 0.15, x: sign * 0.19, z: 0.03 },
    // Lower Thigh (suprapatellar)
    { y: -0.72, rx: 0.12, rz: 0.12, x: sign * 0.19, z: 0.02 },
    // Knee joint / Patella
    { y: -0.92, rx: 0.11, rz: 0.12, x: sign * 0.19, z: 0.04 },
    // Below knee / Tibial tuberosity
    { y: -1.08, rx: 0.11, rz: 0.11, x: sign * 0.19, z: 0.02 },
    // Upper Calf (gastrocnemius muscular bulge posterior)
    { y: -1.26, rx: 0.12, rz: 0.13, x: sign * 0.19, z: -0.01 },
    // Mid Calf
    { y: -1.45, rx: 0.10, rz: 0.11, x: sign * 0.19, z: -0.01 },
    // Lower Calf (achilles tendon narrowing)
    { y: -1.68, rx: 0.075, rz: 0.08, x: sign * 0.19, z: 0.0 },
    // Ankle (medial & lateral malleoli)
    { y: -1.85, rx: 0.075, rz: 0.085, x: sign * 0.19, z: 0.01 },
  ];

  return createLoftedRingGeometry(rings, 24);
}

/**
 * Anatomical Foot Surface: Heel, arch, ball, toes
 */
export function createFootSurfaceGeometry(isLeft = true) {
  const sign = isLeft ? -1 : 1;
  const rings = [
    // Ankle base
    { y: -1.85, rx: 0.07, rz: 0.08, x: sign * 0.19, z: 0.01 },
    // Mid foot / longitudinal arch
    { y: -1.92, rx: 0.08, rz: 0.14, x: sign * 0.19, z: 0.06 },
    // Ball of foot / Metatarsals
    { y: -1.98, rx: 0.085, rz: 0.18, x: sign * 0.19, z: 0.09 },
    // Toes tip
    { y: -2.01, rx: 0.06, rz: 0.10, x: sign * 0.19, z: 0.15 },
  ];

  return createLoftedRingGeometry(rings, 20);
}

/**
 * Anatomical Arm Surface: Shoulder deltoid, Upper arm, Elbow, Forearm, Wrist
 */
export function createArmSurfaceGeometry(isLeft = true) {
  const sign = isLeft ? -1 : 1;
  // Natural slight anatomical abduction (A-pose ~18 degrees)
  const rings = [
    // Deltoid shoulder cap
    { y: 1.15, x: sign * 0.42, z: 0.01, rx: 0.12, rz: 0.12 },
    // Upper Arm (biceps/triceps)
    { y: 0.95, x: sign * 0.52, z: 0.01, rx: 0.095, rz: 0.095 },
    // Mid Upper Arm
    { y: 0.72, x: sign * 0.62, z: 0.01, rx: 0.085, rz: 0.085 },
    // Elbow joint (olecranon posterior)
    { y: 0.52, x: sign * 0.70, z: 0.01, rx: 0.08, rz: 0.08 },
    // Proximal Forearm (brachioradialis / flexor mass)
    { y: 0.34, x: sign * 0.78, z: 0.02, rx: 0.08, rz: 0.075 },
    // Mid Forearm
    { y: 0.15, x: sign * 0.85, z: 0.03, rx: 0.065, rz: 0.065 },
    // Wrist
    { y: -0.02, x: sign * 0.91, z: 0.04, rx: 0.055, rz: 0.05 },
  ];

  return createLoftedRingGeometry(rings, 24);
}

/**
 * Anatomical Hand Surface with 5 distinct articulated fingers
 */
export function createHandSurfaceGroup(isLeft = true, skinMaterial, boneMaterial) {
  const group = new THREE.Group();
  const sign = isLeft ? -1 : 1;
  const palmX = sign * 0.94;
  const palmY = -0.12;
  const palmZ = 0.05;

  // Palm base mesh
  const palmGeo = new THREE.BoxGeometry(0.09, 0.14, 0.035);
  const palmMesh = new THREE.Mesh(palmGeo, skinMaterial);
  palmMesh.position.set(palmX, palmY, palmZ);
  palmMesh.rotation.z = sign * -0.22;
  group.add(palmMesh);

  // Five distinct fingers: Thumb, Index, Middle, Ring, Pinky
  const fingerConfigs = [
    // Thumb: abducted laterally and slightly anteriorly
    { name: 'thumb', ox: sign * 0.04, oy: 0.04, oz: 0.02, len: 0.08, rotZ: sign * 0.65, thick: 0.018 },
    // Index finger
    { name: 'index', ox: sign * 0.02, oy: -0.08, oz: 0.005, len: 0.10, rotZ: sign * 0.05, thick: 0.016 },
    // Middle finger (longest)
    { name: 'middle', ox: 0, oy: -0.09, oz: 0.0, len: 0.11, rotZ: 0, thick: 0.016 },

    // Ring finger
    { name: 'ring', ox: sign * -0.02, oy: -0.085, oz: -0.005, len: 0.10, rotZ: sign * -0.05, thick: 0.015 },
    // Pinky finger
    { name: 'pinky', ox: sign * -0.04, oy: -0.075, oz: -0.01, len: 0.08, rotZ: sign * -0.12, thick: 0.014 },
  ];

  fingerConfigs.forEach((cfg) => {
    // Outer finger skin
    const fSkinGeo = new THREE.CylinderGeometry(cfg.thick * 0.75, cfg.thick, cfg.len, 10);
    const fSkinMesh = new THREE.Mesh(fSkinGeo, skinMaterial);
    fSkinMesh.position.set(palmX + cfg.ox, palmY + cfg.oy - cfg.len * 0.45, palmZ + cfg.oz);
    fSkinMesh.rotation.z = sign * -0.22 + cfg.rotZ;
    group.add(fSkinMesh);

    // Inner finger bone phalanges
    if (boneMaterial) {
      const fBoneGeo = new THREE.CylinderGeometry(cfg.thick * 0.4, cfg.thick * 0.45, cfg.len * 0.9, 8);
      const fBoneMesh = new THREE.Mesh(fBoneGeo, boneMaterial);
      fBoneMesh.position.copy(fSkinMesh.position);
      fBoneMesh.rotation.copy(fSkinMesh.rotation);
      group.add(fBoneMesh);
    }
  });

  return group;
}

/**
 * Anatomical 3D Skull & Facial Skeleton
 * Cranium, eye sockets (orbitals), nasal aperture, zygomatic cheekbones, mandible with chin
 */
export function createDetailedSkullGroup(boneMaterial) {
  const skullGroup = new THREE.Group();
  skullGroup.position.set(0, 1.68, 0);

  // 1. Cranial Vault (Braincase) - anatomically shaped oval with parietal width
  const craniumGeo = new THREE.SphereGeometry(0.24, 24, 24);
  craniumGeo.scale(1.0, 1.15, 1.25);
  const craniumMesh = new THREE.Mesh(craniumGeo, boneMaterial);
  craniumMesh.position.set(0, 0.08, -0.02);
  skullGroup.add(craniumMesh);

  // 2. Facial Skeleton & Maxilla (Upper jaw & nose base)
  const faceGeo = new THREE.BoxGeometry(0.20, 0.18, 0.16);
  const faceMesh = new THREE.Mesh(faceGeo, boneMaterial);
  faceMesh.position.set(0, -0.12, 0.11);
  skullGroup.add(faceMesh);

  // 3. Bilateral Zygomatic Arches (Cheekbones)
  const leftCheek = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.05, 0.14), boneMaterial);
  leftCheek.position.set(-0.16, -0.08, 0.08);
  skullGroup.add(leftCheek);

  const rightCheek = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.05, 0.14), boneMaterial);
  rightCheek.position.set(0.16, -0.08, 0.08);
  skullGroup.add(rightCheek);

  // 4. Eye Orbital Cavities (Hollow anatomical sockets)
  const socketGeo = new THREE.SphereGeometry(0.048, 14, 14);
  const socketMat = new THREE.MeshBasicMaterial({ color: '#010612' });

  const leftEye = new THREE.Mesh(socketGeo, socketMat);
  leftEye.position.set(-0.075, -0.04, 0.19);
  skullGroup.add(leftEye);

  const rightEye = new THREE.Mesh(socketGeo, socketMat);
  rightEye.position.set(0.075, -0.04, 0.19);
  skullGroup.add(rightEye);

  // 5. Nasal Aperture (Piriform aperture)
  const nasalGeo = new THREE.ConeGeometry(0.025, 0.06, 6);
  const nasalMesh = new THREE.Mesh(nasalGeo, socketMat);
  nasalMesh.rotation.x = Math.PI;
  nasalMesh.position.set(0, -0.11, 0.20);
  skullGroup.add(nasalMesh);

  // 6. Mandible (Lower jawbone with chin & ramus)
  const jawCurve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(-0.11, -0.14, 0.02),
    new THREE.Vector3(0, -0.26, 0.21), // Mental protuberance (chin)
    new THREE.Vector3(0.11, -0.14, 0.02)
  );
  const jawGeo = new THREE.TubeGeometry(jawCurve, 20, 0.025, 10, false);
  const jawMesh = new THREE.Mesh(jawGeo, boneMaterial);
  skullGroup.add(jawMesh);

  return skullGroup;
}

/**
 * Anatomical Vertebral Column (Spine)
 * 24 Segmented vertebrae + Intervertebral Discs + Sacrum & Coccyx
 * Respects Cervical lordosis, Thoracic kyphosis, Lumbar lordosis
 */
export function createVertebralColumnGroup(boneMaterial, discMaterial) {
  const spineGroup = new THREE.Group();

  // 24 Vertebrae levels
  const totalVertebrae = 24;
  for (let i = 0; i < totalVertebrae; i++) {
    const t = i / totalVertebrae;
    // Y position from neck (C1 = 1.48) to lower lumbar (L5 = 0.08)
    const y = 1.48 - t * 1.40;

    // Anatomical spinal curvature
    let curveZ = 0;
    if (i < 7) {
      // Cervical lordosis (curves forward)
      curveZ = Math.sin((i / 7) * Math.PI) * 0.025 - 0.02;
    } else if (i < 19) {
      // Thoracic kyphosis (curves backward)
      curveZ = -Math.sin(((i - 7) / 12) * Math.PI) * 0.045 - 0.02;
    } else {
      // Lumbar lordosis (curves forward strongly)
      curveZ = Math.sin(((i - 19) / 5) * Math.PI) * 0.04 - 0.01;
    }

    // Scale vertebra: cervical are smaller, lumbar are larger
    const width = 0.05 + t * 0.035;
    const depth = 0.05 + t * 0.035;
    const height = 0.045;

    // Vertebral Body
    const vertGeo = new THREE.CylinderGeometry(width, width * 1.05, height, 14);
    const vertMesh = new THREE.Mesh(vertGeo, boneMaterial);
    vertMesh.position.set(0, y, curveZ);
    vertMesh.rotation.x = Math.sin(t * Math.PI) * 0.08;
    spineGroup.add(vertMesh);

    // Posterior Spinous Process (the prominent bumps felt down the back)
    const spinousGeo = new THREE.ConeGeometry(0.015, 0.045, 6);
    const spinousMesh = new THREE.Mesh(spinousGeo, boneMaterial);
    spinousMesh.rotation.x = Math.PI / 2 + 0.3;
    spinousMesh.position.set(0, y, curveZ - depth * 0.95);
    spineGroup.add(spinousMesh);

    // Bilateral Transverse Processes
    const transGeo = new THREE.BoxGeometry(width * 2.2, 0.018, 0.02);
    const transMesh = new THREE.Mesh(transGeo, boneMaterial);
    transMesh.position.set(0, y, curveZ - depth * 0.3);
    spineGroup.add(transMesh);

    // Intervertebral Disc
    if (i < totalVertebrae - 1 && discMaterial) {
      const discGeo = new THREE.CylinderGeometry(width * 0.95, width * 0.95, 0.016, 12);
      const discMesh = new THREE.Mesh(discGeo, discMaterial);
      discMesh.position.set(0, y - height * 0.6, curveZ);
      spineGroup.add(discMesh);
    }
  }

  // Sacrum & Coccyx (Triangular fused bone at base of spine)
  const sacrumGeo = new THREE.ConeGeometry(0.12, 0.26, 8);
  const sacrumMesh = new THREE.Mesh(sacrumGeo, boneMaterial);
  sacrumMesh.rotation.x = Math.PI - 0.25; // Tilted slightly posteriorly
  sacrumMesh.position.set(0, -0.05, -0.03);
  sacrumGeo.scale(1.2, 1.0, 0.5);
  spineGroup.add(sacrumMesh);

  return spineGroup;
}

/**
 * Anatomical Thoracic Cage (Ribs, Sternum, Clavicles, Scapulae)
 * 12 Pairs of anatomically accurate ribs wrapping around from vertebrae
 */
export function createThoracicCageGroup(boneMaterial) {
  const cageGroup = new THREE.Group();

  // 1. Sternum (Breastbone: Manubrium, Body, Xiphoid process)
  // Manubrium (upper shield)
  const manubrium = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.09, 0.025), boneMaterial);
  manubrium.position.set(0, 1.12, 0.26);
  cageGroup.add(manubrium);

  // Sternal Body
  const sternalBody = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.28, 0.02), boneMaterial);
  sternalBody.position.set(0, 0.94, 0.28);
  cageGroup.add(sternalBody);

  // Xiphoid Process
  const xiphoid = new THREE.Mesh(new THREE.ConeGeometry(0.025, 0.05, 5), boneMaterial);
  xiphoid.position.set(0, 0.77, 0.27);
  xiphoid.rotation.x = Math.PI;
  cageGroup.add(xiphoid);

  // 2. Clavicles (Collar bones) with subtle S-curve
  const leftClavicleCurve = new THREE.CubicBezierCurve3(
    new THREE.Vector3(-0.05, 1.15, 0.24),
    new THREE.Vector3(-0.18, 1.20, 0.22),
    new THREE.Vector3(-0.32, 1.18, 0.12),
    new THREE.Vector3(-0.42, 1.15, 0.02) // Acromion joint
  );
  const leftClavMesh = new THREE.Mesh(new THREE.TubeGeometry(leftClavicleCurve, 16, 0.022, 8, false), boneMaterial);
  cageGroup.add(leftClavMesh);

  const rightClavicleCurve = new THREE.CubicBezierCurve3(
    new THREE.Vector3(0.05, 1.15, 0.24),
    new THREE.Vector3(0.18, 1.20, 0.22),
    new THREE.Vector3(0.32, 1.18, 0.12),
    new THREE.Vector3(0.42, 1.15, 0.02)
  );
  const rightClavMesh = new THREE.Mesh(new THREE.TubeGeometry(rightClavicleCurve, 16, 0.022, 8, false), boneMaterial);
  cageGroup.add(rightClavMesh);

  // 3. Scapulae (Shoulder blades, Posterior thorax)
  const leftScapula = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.32, 0.025), boneMaterial);
  leftScapula.position.set(-0.28, 0.98, -0.17);
  leftScapula.rotation.set(0.15, 0.25, -0.1);
  cageGroup.add(leftScapula);

  const rightScapula = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.32, 0.025), boneMaterial);
  rightScapula.position.set(0.28, 0.98, -0.17);
  rightScapula.rotation.set(0.15, -0.25, 0.1);
  cageGroup.add(rightScapula);

  // 4. 12 Pairs of Anatomical Ribs
  for (let i = 0; i < 12; i++) {
    // Rib height from upper thorax (T1) down to costal margin
    const ribY = 1.12 - i * 0.048;
    // Width increases from 1 to 7 then decreases slightly
    const spanFactor = i < 7 ? 0.28 + i * 0.022 : 0.42 - (i - 7) * 0.02;
    const ribThick = 0.016;

    // Left Rib Curve: from spine posterior around to sternum anterior
    const leftCurve = new THREE.CubicBezierCurve3(
      new THREE.Vector3(-0.04, ribY + 0.02, -0.06 - (i > 3 ? 0.03 : 0)), // Spine attachment
      new THREE.Vector3(-spanFactor * 1.05, ribY + 0.01, -0.08),          // Lateral flare
      new THREE.Vector3(-spanFactor * 0.95, ribY - 0.04, 0.18),          // Anterior turn
      new THREE.Vector3(i < 8 ? -0.05 : -0.12, ribY - 0.05, 0.26)        // Sternum / costal cartilage
    );
    const leftRib = new THREE.Mesh(new THREE.TubeGeometry(leftCurve, 18, ribThick, 8, false), boneMaterial);
    cageGroup.add(leftRib);

    // Right Rib Curve
    const rightCurve = new THREE.CubicBezierCurve3(
      new THREE.Vector3(0.04, ribY + 0.02, -0.06 - (i > 3 ? 0.03 : 0)),
      new THREE.Vector3(spanFactor * 1.05, ribY + 0.01, -0.08),
      new THREE.Vector3(spanFactor * 0.95, ribY - 0.04, 0.18),
      new THREE.Vector3(i < 8 ? 0.05 : 0.12, ribY - 0.05, 0.26)
    );
    const rightRib = new THREE.Mesh(new THREE.TubeGeometry(rightCurve, 18, ribThick, 8, false), boneMaterial);
    cageGroup.add(rightRib);
  }

  return cageGroup;
}

/**
 * Anatomical Pelvis (Iliac wings, pubic symphysis, ischial tuberosities)
 */
export function createPelvisGroup(boneMaterial) {
  const pelvisGroup = new THREE.Group();
  pelvisGroup.position.set(0, -0.02, 0.01);

  // Left Ilium (Wing)
  const leftIliumGeo = new THREE.TorusGeometry(0.19, 0.042, 10, 24, Math.PI * 0.85);
  const leftIlium = new THREE.Mesh(leftIliumGeo, boneMaterial);
  leftIlium.position.set(-0.21, 0.06, 0.01);
  leftIlium.rotation.set(0.15, 0.35, -0.4);
  pelvisGroup.add(leftIlium);

  // Right Ilium (Wing)
  const rightIliumGeo = new THREE.TorusGeometry(0.19, 0.042, 10, 24, Math.PI * 0.85);
  const rightIlium = new THREE.Mesh(rightIliumGeo, boneMaterial);
  rightIlium.position.set(0.21, 0.06, 0.01);
  rightIlium.rotation.set(0.15, -0.35, 0.4);
  pelvisGroup.add(rightIlium);

  // Pubic Arch & Ischium base
  const pubicGeo = new THREE.TorusGeometry(0.14, 0.035, 8, 18, Math.PI);
  const pubicMesh = new THREE.Mesh(pubicGeo, boneMaterial);
  pubicMesh.position.set(0, -0.16, 0.08);
  pubicMesh.rotation.set(Math.PI, 0, 0);
  pelvisGroup.add(pubicMesh);

  return pelvisGroup;
}

/**
 * Appendicular Skeleton: Long bones (Femur, Tibia, Fibula, Patella, Humerus, Radius, Ulna)
 */
export function createAppendicularSkeletonGroup(boneMaterial) {
  const group = new THREE.Group();

  [-1, 1].forEach((sign) => {
    // 1. Femur (Thigh bone) - angled femoral neck + head into acetabulum
    const femurGroup = new THREE.Group();
    femurGroup.position.set(sign * 0.19, -0.22, 0.02);

    // Femoral Head & Neck
    const headGeo = new THREE.SphereGeometry(0.048, 12, 12);
    const headMesh = new THREE.Mesh(headGeo, boneMaterial);
    headMesh.position.set(sign * -0.04, 0.04, 0);
    femurGroup.add(headMesh);

    // Greater Trochanter
    const trochMesh = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.05, 0.04), boneMaterial);
    trochMesh.position.set(sign * 0.04, 0.02, 0);
    femurGroup.add(trochMesh);

    // Femoral Shaft
    const shaftGeo = new THREE.CylinderGeometry(0.04, 0.045, 0.68, 14);
    const shaftMesh = new THREE.Mesh(shaftGeo, boneMaterial);
    shaftMesh.position.set(0, -0.34, 0);
    shaftMesh.rotation.z = sign * -0.035; // Natural valgus angle
    femurGroup.add(shaftMesh);

    // Femoral Condyles (Knee joint)
    const condyleGeo = new THREE.BoxGeometry(0.09, 0.05, 0.07);
    const condyleMesh = new THREE.Mesh(condyleGeo, boneMaterial);
    condyleMesh.position.set(0, -0.68, 0.01);
    femurGroup.add(condyleMesh);

    group.add(femurGroup);

    // 2. Patella (Kneecap)
    const patellaGeo = new THREE.SphereGeometry(0.038, 10, 10);
    patellaGeo.scale(1.0, 1.2, 0.5);
    const patellaMesh = new THREE.Mesh(patellaGeo, boneMaterial);
    patellaMesh.position.set(sign * 0.19, -0.92, 0.07);
    group.add(patellaMesh);

    // 3. Lower Leg: Tibia (shin) & Fibula (slender lateral)
    const tibiaGeo = new THREE.CylinderGeometry(0.044, 0.035, 0.82, 14);
    const tibiaMesh = new THREE.Mesh(tibiaGeo, boneMaterial);
    tibiaMesh.position.set(sign * 0.18, -1.40, 0.02);
    group.add(tibiaMesh);

    const fibulaGeo = new THREE.CylinderGeometry(0.018, 0.016, 0.80, 8);
    const fibulaMesh = new THREE.Mesh(fibulaGeo, boneMaterial);
    fibulaMesh.position.set(sign * 0.24, -1.40, 0.0);
    group.add(fibulaMesh);

    // 4. Arm: Humerus (Upper arm bone)
    const humerusGeo = new THREE.CylinderGeometry(0.034, 0.036, 0.56, 12);
    const humerusMesh = new THREE.Mesh(humerusGeo, boneMaterial);
    humerusMesh.position.set(sign * 0.56, 0.85, 0.01);
    humerusMesh.rotation.z = sign * 0.22;
    group.add(humerusMesh);

    // 5. Forearm: Radius & Ulna
    const radiusGeo = new THREE.CylinderGeometry(0.024, 0.026, 0.48, 10);
    const radiusMesh = new THREE.Mesh(radiusGeo, boneMaterial);
    radiusMesh.position.set(sign * 0.78, 0.35, 0.02);
    radiusMesh.rotation.z = sign * 0.24;
    group.add(radiusMesh);

    const ulnaGeo = new THREE.CylinderGeometry(0.022, 0.022, 0.48, 10);
    const ulnaMesh = new THREE.Mesh(ulnaGeo, boneMaterial);
    ulnaMesh.position.set(sign * 0.82, 0.35, 0.0);
    ulnaMesh.rotation.z = sign * 0.24;
    group.add(ulnaMesh);
  });

  return group;
}

/**
 * Major Internal Organs (Lungs with lobes, Heart & Great Vessels, Liver, Stomach, Kidneys, Intestines)
 */
export function createMajorOrgansGroup(organMaterials) {
  const organsGroup = new THREE.Group();

  // 1. Bilateral Lungs with Anatomical Lobes
  // Right Lung (3 Lobes: Superior, Middle, Inferior)
  const rightLungGroup = new THREE.Group();
  rightLungGroup.position.set(0.20, 0.94, 0.05);

  const rSupLobe = new THREE.Mesh(new THREE.SphereGeometry(0.14, 16, 16), organMaterials.lung);
  rSupLobe.position.set(0, 0.10, 0);
  rSupLobe.scale.set(1.0, 1.25, 0.85);
  rightLungGroup.add(rSupLobe);

  const rMidLobe = new THREE.Mesh(new THREE.SphereGeometry(0.13, 14, 14), organMaterials.lung);
  rMidLobe.position.set(0.02, -0.04, 0.02);
  rMidLobe.scale.set(1.05, 0.9, 0.9);
  rightLungGroup.add(rMidLobe);

  const rInfLobe = new THREE.Mesh(new THREE.SphereGeometry(0.15, 16, 16), organMaterials.lung);
  rInfLobe.position.set(0, -0.16, -0.02);
  rInfLobe.scale.set(1.1, 1.1, 0.95);
  rightLungGroup.add(rInfLobe);

  organsGroup.add(rightLungGroup);

  // Left Lung (2 Lobes: Superior, Inferior with cardiac notch)
  const leftLungGroup = new THREE.Group();
  leftLungGroup.position.set(-0.20, 0.94, 0.05);

  const lSupLobe = new THREE.Mesh(new THREE.SphereGeometry(0.14, 16, 16), organMaterials.lung);
  lSupLobe.position.set(0, 0.08, 0);
  lSupLobe.scale.set(0.95, 1.3, 0.85);
  leftLungGroup.add(lSupLobe);

  const lInfLobe = new THREE.Mesh(new THREE.SphereGeometry(0.14, 16, 16), organMaterials.lung);
  lInfLobe.position.set(-0.01, -0.15, -0.02);
  lInfLobe.scale.set(1.0, 1.1, 0.9);
  leftLungGroup.add(lInfLobe);

  organsGroup.add(leftLungGroup);

  // 2. Heart (Cardiac cone in middle mediastinum with leftward apex tilt)
  const heartGroup = new THREE.Group();
  heartGroup.position.set(-0.06, 0.92, 0.12);

  const heartGeo = new THREE.SphereGeometry(0.12, 16, 16);
  heartGeo.scale(1.0, 1.25, 0.9);
  const heartMesh = new THREE.Mesh(heartGeo, organMaterials.heart);
  heartMesh.rotation.set(0.25, 0, 0.35); // Anatomical left-inferior apex orientation
  heartGroup.add(heartMesh);

  // Aorta Arch & Pulmonary Trunk
  const aortaCurve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(0, 0.08, 0),
    new THREE.Vector3(0.04, 0.22, 0.02),
    new THREE.Vector3(-0.02, 0.18, -0.06)
  );
  const aortaMesh = new THREE.Mesh(new THREE.TubeGeometry(aortaCurve, 14, 0.028, 8, false), organMaterials.vessels);
  heartGroup.add(aortaMesh);

  organsGroup.add(heartGroup);

  // 3. Liver (Large wedge-shaped organ in right upper abdomen under diaphragm)
  const liverGeo = new THREE.SphereGeometry(0.21, 18, 18);
  liverGeo.scale(1.4, 0.95, 0.85);
  const liverMesh = new THREE.Mesh(liverGeo, organMaterials.liver);
  liverMesh.position.set(0.12, 0.62, 0.07);
  liverMesh.rotation.set(-0.15, 0.1, -0.2);
  organsGroup.add(liverMesh);

  // 4. Stomach (J-shaped digestive organ in left upper quadrant)
  const stomachCurve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(-0.05, 0.72, 0.06),
    new THREE.Vector3(-0.18, 0.60, 0.08),
    new THREE.Vector3(-0.08, 0.50, 0.09)
  );
  const stomachMesh = new THREE.Mesh(new THREE.TubeGeometry(stomachCurve, 16, 0.065, 10, false), organMaterials.stomach);
  organsGroup.add(stomachMesh);

  // 5. Bilateral Kidneys (Retroperitoneal bean-shaped organs)
  const kidneyGeo = new THREE.SphereGeometry(0.065, 12, 12);
  kidneyGeo.scale(0.8, 1.3, 0.7);

  const leftKidney = new THREE.Mesh(kidneyGeo, organMaterials.kidney);
  leftKidney.position.set(-0.16, 0.44, -0.06);
  organsGroup.add(leftKidney);

  const rightKidney = new THREE.Mesh(kidneyGeo, organMaterials.kidney);
  rightKidney.position.set(0.16, 0.40, -0.06); // Slightly lower due to liver
  organsGroup.add(rightKidney);

  // 6. Intestines (Colon frame + central small bowel loops)
  // Large Intestine / Colon Frame (Ascending, Transverse, Descending)
  const colonCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(0.19, 0.06, 0.08),  // Cecum / Appendix
    new THREE.Vector3(0.20, 0.36, 0.08),  // Ascending colon
    new THREE.Vector3(0.14, 0.45, 0.09),  // Hepatic flexure
    new THREE.Vector3(0.0, 0.42, 0.12),   // Transverse colon
    new THREE.Vector3(-0.15, 0.46, 0.09), // Splenic flexure
    new THREE.Vector3(-0.19, 0.28, 0.08), // Descending colon
    new THREE.Vector3(-0.14, 0.05, 0.07), // Sigmoid colon
    new THREE.Vector3(0.0, -0.08, 0.04),  // Rectum
  ]);
  const colonMesh = new THREE.Mesh(new THREE.TubeGeometry(colonCurve, 32, 0.038, 10, false), organMaterials.intestine);
  organsGroup.add(colonMesh);

  // Small Intestine (Dense central coils in umbilical region)
  const smallBowel = new THREE.Mesh(new THREE.TorusGeometry(0.12, 0.055, 12, 24), organMaterials.intestine);
  smallBowel.position.set(0.01, 0.22, 0.09);
  smallBowel.scale.set(1.1, 0.9, 0.8);
  organsGroup.add(smallBowel);

  return organsGroup;
}

/**
 * Major Cardiovascular Vascular Tree (Aorta, Vena Cava, Carotids, Femoral Arteries)
 */
export function createVascularTreeGroup(vascularMaterial) {
  const group = new THREE.Group();

  // Descending Thoracic & Abdominal Aorta
  const aortaPath = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.02, 1.08, 0.02),
    new THREE.Vector3(-0.02, 0.72, -0.02),
    new THREE.Vector3(-0.01, 0.38, -0.01),
    new THREE.Vector3(0.0, 0.12, 0.01),    // Aortic bifurcation
  ]);
  const aortaMesh = new THREE.Mesh(new THREE.TubeGeometry(aortaPath, 20, 0.016, 8, false), vascularMaterial);
  group.add(aortaMesh);

  // Common Iliac & Femoral Arteries (Bilateral legs)
  [-1, 1].forEach((sign) => {
    const iliacCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.0, 0.12, 0.01),
      new THREE.Vector3(sign * 0.09, -0.04, 0.02),
      new THREE.Vector3(sign * 0.16, -0.22, 0.03),
      new THREE.Vector3(sign * 0.18, -0.65, 0.02), // Femoral artery down thigh
      new THREE.Vector3(sign * 0.18, -1.15, 0.01), // Popliteal & Tibial down calf
    ]);
    const iliacMesh = new THREE.Mesh(new THREE.TubeGeometry(iliacCurve, 24, 0.012, 6, false), vascularMaterial);
    group.add(iliacMesh);

    // Carotid Artery (Upward into neck & head)
    const carotidCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(sign * 0.04, 1.15, 0.04),
      new THREE.Vector3(sign * 0.06, 1.35, 0.03),
      new THREE.Vector3(sign * 0.08, 1.55, 0.04),
    ]);
    const carotidMesh = new THREE.Mesh(new THREE.TubeGeometry(carotidCurve, 12, 0.010, 6, false), vascularMaterial);
    group.add(carotidMesh);

    // Subclavian & Brachial Artery (Outward into arm)
    const brachialCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(sign * 0.08, 1.14, 0.03),
      new THREE.Vector3(sign * 0.32, 1.12, 0.02),
      new THREE.Vector3(sign * 0.54, 0.85, 0.01),
      new THREE.Vector3(sign * 0.74, 0.35, 0.02),
    ]);
    const brachialMesh = new THREE.Mesh(new THREE.TubeGeometry(brachialCurve, 16, 0.011, 6, false), vascularMaterial);
    group.add(brachialMesh);
  });

  return group;
}
