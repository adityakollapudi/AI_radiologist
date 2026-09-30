import * as THREE from 'three';

/**
 * Custom Holographic Fresnel Shader for semi-transparent medical visualization
 * Creates bright glowing cyan contours while keeping the internal volume
 * transparent so internal skeleton and organs are clearly visible through the skin.
 */
export const HologramSkinShader = {
  uniforms: {
    uTime: { value: 0 },
    uColor: { value: new THREE.Color('#00e5ff') },
    uDeepColor: { value: new THREE.Color('#03122b') },
    uRimPower: { value: 2.8 },
    uOpacity: { value: 0.85 },
    uScanY: { value: 0.0 },
    uHoverBoost: { value: 0.0 },
  },
  vertexShader: `
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    varying vec3 vWorldPosition;

    void main() {
      vNormal = normalize(normalMatrix * normal);
      vec4 worldPos = modelMatrix * vec4(position, 1.0);
      vWorldPosition = worldPos.xyz;
      vec4 mvPosition = viewMatrix * worldPos;
      vViewPosition = -mvPosition.xyz;
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: `
    uniform float uTime;
    uniform vec3 uColor;
    uniform vec3 uDeepColor;
    uniform float uRimPower;
    uniform float uOpacity;
    uniform float uScanY;
    uniform float uHoverBoost;

    varying vec3 vNormal;
    varying vec3 vViewPosition;
    varying vec3 vWorldPosition;

    void main() {
      vec3 normal = normalize(vNormal);
      vec3 viewDir = normalize(vViewPosition);

      // Fresnel rim glow calculation
      float dotNV = clamp(dot(normal, viewDir), 0.0, 1.0);
      float fresnel = pow(1.0 - dotNV, uRimPower);

      // Subtle horizontal medical scan line pattern
      float scanPattern = sin(vWorldPosition.y * 36.0 + uTime * 1.5) * 0.06 + 0.94;

      // Laser scan beam proximity glow
      float distToBeam = abs(vWorldPosition.y - uScanY);
      float beamGlow = exp(-distToBeam * 7.0) * 0.6;

      // Combine colors
      vec3 finalColor = mix(uDeepColor, uColor, fresnel * scanPattern);
      finalColor += vec3(0.0, 0.9, 1.0) * beamGlow;
      finalColor += uColor * (uHoverBoost * 0.35);

      // Transparency: high at rim/edges, subtle in front-facing center
      float alpha = clamp(fresnel * uOpacity + 0.12 + beamGlow * 0.35, 0.0, 0.95);

      gl_FragColor = vec4(finalColor, alpha);
    }
  `,
};

/**
 * Organ Shader: Internal volumetric glow with soft gradient
 */
export const OrganGlowShader = {
  uniforms: {
    uTime: { value: 0 },
    uBaseColor: { value: new THREE.Color('#06b6d4') },
    uEmissiveColor: { value: new THREE.Color('#00f2ff') },
    uPulse: { value: 1.0 },
    uOpacity: { value: 0.65 },
  },
  vertexShader: `
    varying vec3 vNormal;
    varying vec3 vViewPosition;

    void main() {
      vNormal = normalize(normalMatrix * normal);
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      vViewPosition = -mvPosition.xyz;
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: `
    uniform vec3 uBaseColor;
    uniform vec3 uEmissiveColor;
    uniform float uPulse;
    uniform float uOpacity;

    varying vec3 vNormal;
    varying vec3 vViewPosition;

    void main() {
      vec3 normal = normalize(vNormal);
      vec3 viewDir = normalize(vViewPosition);
      float dotNV = clamp(dot(normal, viewDir), 0.0, 1.0);
      float rim = pow(1.0 - dotNV, 1.8);

      vec3 color = mix(uBaseColor, uEmissiveColor, rim * 0.7) * uPulse;
      gl_FragColor = vec4(color, uOpacity);
    }
  `,
};
