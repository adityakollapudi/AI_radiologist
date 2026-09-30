/**
 * Helper to generate realistic medical scan data & Grad-CAM overlays
 * client-side into base64 PNG data URLs.
 */

export function generateMedicalCanvasImages(modality, subtype = 'default') {
  if (typeof window === 'undefined') return { original: '', gradcam: '' };

  const size = 320;
  const originalCanvas = document.createElement('canvas');
  originalCanvas.width = size;
  originalCanvas.height = size;
  const oCtx = originalCanvas.getContext('2d');

  const gradcamCanvas = document.createElement('canvas');
  gradcamCanvas.width = size;
  gradcamCanvas.height = size;
  const gCtx = gradcamCanvas.getContext('2d');

  // Background
  oCtx.fillStyle = '#060a12';
  oCtx.fillRect(0, 0, size, size);

  if (modality === 'xray') {
    if (subtype === 'musculoskeletal') {
      // Musculoskeletal / Hand or Arm bone X-ray
      oCtx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      oCtx.lineWidth = 1;
      // Soft tissue silhouette
      oCtx.fillStyle = 'rgba(180, 200, 220, 0.12)';
      oCtx.beginPath();
      oCtx.roundRect(size * 0.35, size * 0.1, size * 0.3, size * 0.85, 20);
      oCtx.fill();

      // Radius & Ulna bones
      oCtx.fillStyle = 'rgba(240, 245, 255, 0.75)';
      oCtx.beginPath();
      oCtx.roundRect(size * 0.40, size * 0.15, size * 0.09, size * 0.7, 8);
      oCtx.roundRect(size * 0.51, size * 0.15, size * 0.08, size * 0.7, 8);
      oCtx.fill();

      // Joint articulation & fracture fissure
      oCtx.strokeStyle = '#000000';
      oCtx.lineWidth = 2.5;
      oCtx.beginPath();
      oCtx.moveTo(size * 0.40, size * 0.42);
      oCtx.lineTo(size * 0.49, size * 0.46);
      oCtx.stroke();

      // Grad-CAM attention at fracture site
      const grad = gCtx.createRadialGradient(size * 0.45, size * 0.44, 5, size * 0.45, size * 0.44, size * 0.28);
      grad.addColorStop(0, 'rgba(255, 30, 30, 0.95)');
      grad.addColorStop(0.35, 'rgba(255, 170, 0, 0.8)');
      grad.addColorStop(0.65, 'rgba(0, 210, 255, 0.5)');
      grad.addColorStop(1, 'rgba(0, 0, 50, 0)');
      gCtx.fillStyle = grad;
      gCtx.fillRect(0, 0, size, size);

    } else if (subtype === 'normal') {
      // Normal chest
      drawChestXray(oCtx, size, false);
      // Diffuse low attention
      const grad = gCtx.createRadialGradient(size * 0.5, size * 0.5, 10, size * 0.5, size * 0.5, size * 0.35);
      grad.addColorStop(0, 'rgba(0, 180, 255, 0.4)');
      grad.addColorStop(1, 'rgba(0, 0, 40, 0)');
      gCtx.fillStyle = grad;
      gCtx.fillRect(0, 0, size, size);

    } else {
      // Chest X-ray with Pneumonia infiltrates
      drawChestXray(oCtx, size, true);

      // Grad-CAM attention focused on right lower lung lobe infiltrate
      const grad = gCtx.createRadialGradient(size * 0.65, size * 0.55, 8, size * 0.65, size * 0.55, size * 0.32);
      grad.addColorStop(0, 'rgba(255, 20, 20, 0.95)');
      grad.addColorStop(0.3, 'rgba(255, 150, 0, 0.85)');
      grad.addColorStop(0.6, 'rgba(0, 230, 255, 0.6)');
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      gCtx.fillStyle = grad;
      gCtx.fillRect(0, 0, size, size);
    }

  } else if (modality === 'ct') {
    // 2D CT slice: Body cross section with lungs and spine
    // Body contour
    oCtx.fillStyle = 'rgba(100, 115, 140, 0.25)';
    oCtx.beginPath();
    oCtx.ellipse(size * 0.5, size * 0.5, size * 0.42, size * 0.36, 0, 0, Math.PI * 2);
    oCtx.fill();

    // Thoracic wall / ribs
    oCtx.strokeStyle = 'rgba(240, 245, 255, 0.85)';
    oCtx.lineWidth = 4;
    oCtx.stroke();

    // Spine at posterior
    oCtx.fillStyle = 'rgba(250, 250, 255, 0.95)';
    oCtx.beginPath();
    oCtx.arc(size * 0.5, size * 0.72, size * 0.08, 0, Math.PI * 2);
    oCtx.fill();

    // Lung fields (hypodense / dark air)
    oCtx.fillStyle = '#070b14';
    // Left lung field
    oCtx.beginPath();
    oCtx.ellipse(size * 0.35, size * 0.48, size * 0.12, size * 0.16, -0.15, 0, Math.PI * 2);
    oCtx.fill();
    // Right lung field
    oCtx.beginPath();
    oCtx.ellipse(size * 0.65, size * 0.48, size * 0.12, size * 0.16, 0.15, 0, Math.PI * 2);
    oCtx.fill();

    // Ground glass opacities / infiltrates (subtle peripheral haziness)
    oCtx.fillStyle = 'rgba(200, 220, 240, 0.55)';
    oCtx.beginPath();
    oCtx.arc(size * 0.72, size * 0.45, size * 0.06, 0, Math.PI * 2);
    oCtx.arc(size * 0.30, size * 0.52, size * 0.05, 0, Math.PI * 2);
    oCtx.fill();

    // Grad-CAM for COVID-19 peripheral bilateral ground glass
    const grad1 = gCtx.createRadialGradient(size * 0.72, size * 0.45, 4, size * 0.72, size * 0.45, size * 0.22);
    grad1.addColorStop(0, 'rgba(255, 0, 50, 0.95)');
    grad1.addColorStop(0.4, 'rgba(255, 180, 0, 0.85)');
    grad1.addColorStop(0.7, 'rgba(0, 200, 255, 0.5)');
    grad1.addColorStop(1, 'rgba(0, 0, 0, 0)');
    gCtx.fillStyle = grad1;
    gCtx.fillRect(0, 0, size, size);

    const grad2 = gCtx.createRadialGradient(size * 0.30, size * 0.52, 4, size * 0.30, size * 0.52, size * 0.18);
    grad2.addColorStop(0, 'rgba(255, 50, 0, 0.9)');
    grad2.addColorStop(0.5, 'rgba(255, 200, 0, 0.7)');
    grad2.addColorStop(1, 'rgba(0, 0, 0, 0)');
    gCtx.fillStyle = grad2;
    gCtx.fillRect(0, 0, size, size);

  } else if (modality === 'mri') {
    // MRI Brain Axial Slice
    // Calvarium / Skull contour
    oCtx.strokeStyle = 'rgba(240, 240, 255, 0.85)';
    oCtx.lineWidth = 3;
    oCtx.beginPath();
    oCtx.ellipse(size * 0.5, size * 0.5, size * 0.38, size * 0.44, 0, 0, Math.PI * 2);
    oCtx.stroke();

    // Brain parenchyma
    oCtx.fillStyle = 'rgba(120, 135, 160, 0.35)';
    oCtx.beginPath();
    oCtx.ellipse(size * 0.5, size * 0.5, size * 0.35, size * 0.41, 0, 0, Math.PI * 2);
    oCtx.fill();

    // Hemispheric fissure
    oCtx.strokeStyle = 'rgba(0, 0, 0, 0.4)';
    oCtx.lineWidth = 2;
    oCtx.beginPath();
    oCtx.moveTo(size * 0.5, size * 0.15);
    oCtx.lineTo(size * 0.5, size * 0.85);
    oCtx.stroke();

    // Ventricles (butterfly shape)
    oCtx.fillStyle = 'rgba(20, 30, 50, 0.8)';
    oCtx.beginPath();
    oCtx.ellipse(size * 0.46, size * 0.48, size * 0.03, size * 0.08, -0.1, 0, Math.PI * 2);
    oCtx.ellipse(size * 0.54, size * 0.48, size * 0.03, size * 0.08, 0.1, 0, Math.PI * 2);
    oCtx.fill();

    // Hyperintense Tumor mass in left frontal-parietal region
    if (subtype !== 'notumor') {
      oCtx.fillStyle = 'rgba(255, 255, 255, 0.88)';
      oCtx.beginPath();
      oCtx.arc(size * 0.36, size * 0.38, size * 0.08, 0, Math.PI * 2);
      oCtx.fill();
      oCtx.strokeStyle = 'rgba(180, 220, 255, 0.6)';
      oCtx.lineWidth = 3;
      oCtx.stroke();

      // Grad-CAM focused precisely on the tumor mass
      const grad = gCtx.createRadialGradient(size * 0.36, size * 0.38, 5, size * 0.36, size * 0.38, size * 0.22);
      grad.addColorStop(0, 'rgba(255, 0, 0, 0.95)');
      grad.addColorStop(0.35, 'rgba(255, 160, 0, 0.85)');
      grad.addColorStop(0.65, 'rgba(0, 220, 255, 0.5)');
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      gCtx.fillStyle = grad;
      gCtx.fillRect(0, 0, size, size);
    } else {
      // Normal Brain: low uniform response
      const grad = gCtx.createRadialGradient(size * 0.5, size * 0.5, 10, size * 0.5, size * 0.5, size * 0.3);
      grad.addColorStop(0, 'rgba(0, 180, 255, 0.35)');
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      gCtx.fillStyle = grad;
      gCtx.fillRect(0, 0, size, size);
    }
  }

  return {
    original: originalCanvas.toDataURL('image/png'),
    gradcam: gradcamCanvas.toDataURL('image/png'),
  };
}

function drawChestXray(ctx, size, withPneumonia) {
  // Thoracic cage outline
  ctx.strokeStyle = 'rgba(230, 240, 255, 0.65)';
  ctx.lineWidth = 2;

  // Bilateral lung fields
  ctx.fillStyle = 'rgba(10, 15, 25, 0.9)';
  // Left lung
  ctx.beginPath();
  ctx.moveTo(size * 0.32, size * 0.22);
  ctx.bezierCurveTo(size * 0.15, size * 0.32, size * 0.18, size * 0.72, size * 0.35, size * 0.78);
  ctx.bezierCurveTo(size * 0.42, size * 0.78, size * 0.44, size * 0.5, size * 0.44, size * 0.24);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Right lung
  ctx.beginPath();
  ctx.moveTo(size * 0.68, size * 0.22);
  ctx.bezierCurveTo(size * 0.85, size * 0.32, size * 0.82, size * 0.72, size * 0.65, size * 0.78);
  ctx.bezierCurveTo(size * 0.58, size * 0.78, size * 0.56, size * 0.5, size * 0.56, size * 0.24);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Rib shadows across lungs
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
  ctx.lineWidth = 3;
  for (let i = 0; i < 6; i++) {
    const y = size * (0.28 + i * 0.08);
    // Left ribs
    ctx.beginPath();
    ctx.moveTo(size * 0.44, y - 5);
    ctx.quadraticCurveTo(size * 0.32, y + 15, size * 0.20, y + 8);
    ctx.stroke();
    // Right ribs
    ctx.beginPath();
    ctx.moveTo(size * 0.56, y - 5);
    ctx.quadraticCurveTo(size * 0.68, y + 15, size * 0.80, y + 8);
    ctx.stroke();
  }

  // Cardiac silhouette
  ctx.fillStyle = 'rgba(230, 240, 255, 0.45)';
  ctx.beginPath();
  ctx.ellipse(size * 0.52, size * 0.62, size * 0.14, size * 0.12, 0.25, 0, Math.PI * 2);
  ctx.fill();

  // Spine column in midline
  ctx.fillStyle = 'rgba(240, 245, 255, 0.65)';
  ctx.fillRect(size * 0.48, size * 0.15, size * 0.04, size * 0.75);

  if (withPneumonia) {
    // Pneumonia consolidation / infiltrate in right lower lobe
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.beginPath();
    ctx.ellipse(size * 0.66, size * 0.58, size * 0.09, size * 0.07, 0.1, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * Verified Medical Model Details per repository specification
 */
export const VERIFIED_MODELS = {
  xray_chest: {
    name: 'EfficientNet-B3 (Chest Specialist)',
    framework: 'PyTorch 2.2',
    input_size: '224x224 px',
    threshold: 0.75,
    dataset: 'ChestX-ray14 (NIH Clinical Center, 112,120 radiographs)',
    metrics: {
      'ROC-AUC': '0.932',
      'Sensitivity': '89.4%',
      'Specificity': '92.1%',
      'F1-Score': '0.907',
    },
    classes: ['NORMAL', 'PNEUMONIA'],
  },
  xray_mura: {
    name: 'DenseNet-169 (MURA Specialist)',
    framework: 'PyTorch 2.2',
    input_size: '320x320 px',
    threshold: 0.70,
    dataset: 'MURA v1.1 (Stanford Musculoskeletal Radiographs, 40,561 studies)',
    metrics: {
      'ROC-AUC': '0.895',
      'Sensitivity': '86.8%',
      'Specificity': '88.2%',
      "Cohen's Kappa": '0.741',
    },
    classes: ['NORMAL', 'ABNORMAL'],
  },
  ct: {
    name: 'DenseNet-121 (Axial Slice Classifier)',
    framework: 'PyTorch 2.2',
    input_size: '512x512 px (Single 2D Axial Slice)',
    threshold: 0.65,
    dataset: 'COVID-CT / CC-CCII (Consortium for COVID-19 Clinical Image Analysis)',
    metrics: {
      'Accuracy': '89.1%',
      'Sensitivity': '87.5%',
      'Specificity': '91.0%',
      'ROC-AUC': '0.941',
    },
    classes: ['NORMAL', 'PNEUMONIA', 'COVID-19'],
  },
  mri: {
    name: 'ResNet-50 Classifier (Brain MRI)',
    framework: 'PyTorch 2.2',
    input_size: '224x224 px (T1-ce / FLAIR axial sequence)',
    threshold: 0.70,
    dataset: 'BraTS Benchmark (Brain Tumor Segmentation Challenge cohort)',
    metrics: {
      'ROC-AUC': '0.963',
      'Accuracy': '93.8%',
      'Precision': '94.1%',
      'Recall': '93.4%',
    },
    classes: ['NO TUMOR', 'TUMOR'],
  },
};

/**
 * Predefined realistic test sample presets for demo / testing
 */
export const SAMPLE_PRESETS = [
  {
    id: 'xray_chest_pneumonia',
    modality: 'xray',
    label: 'Chest X-Ray (Pneumonia Consolidation)',
    description: 'Bilateral thoracic radiograph showing focal consolidation',
    subtype: 'default',
  },
  {
    id: 'xray_chest_normal',
    modality: 'xray',
    label: 'Chest X-Ray (Normal Thorax)',
    description: 'Clear bilateral lung fields, normal cardiothoracic ratio',
    subtype: 'normal',
  },
  {
    id: 'xray_mura_fracture',
    modality: 'xray',
    label: 'Musculoskeletal X-Ray (MURA Abnormality)',
    description: 'Forearm radiograph with radius/ulna cortical disruption',
    subtype: 'musculoskeletal',
  },
  {
    id: 'ct_covid',
    modality: 'ct',
    label: 'CT Scan Slice (COVID-19 Ground-Glass)',
    description: '2D Axial chest CT slice with peripheral bilateral opacities',
    subtype: 'default',
  },
  {
    id: 'mri_tumor',
    modality: 'mri',
    label: 'Brain MRI (Tumor Mass)',
    description: 'Axial T1ce/FLAIR brain slice with hyperintense lesion',
    subtype: 'default',
  },
  {
    id: 'mri_notumor',
    modality: 'mri',
    label: 'Brain MRI (No Tumor)',
    description: 'Normal brain axial slice, symmetric ventricular system',
    subtype: 'notumor',
  },
];
