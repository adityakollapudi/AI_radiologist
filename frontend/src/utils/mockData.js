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
    if (subtype === 'brain_bleed') {
      // Non-contrast cranial CT slice for Acute Intracranial Hemorrhage / Brain Bleed
      // Calvarium / Cranial bone (dense bone: bright white)
      oCtx.strokeStyle = 'rgba(250, 250, 255, 0.95)';
      oCtx.lineWidth = 6;
      oCtx.beginPath();
      oCtx.ellipse(size * 0.5, size * 0.5, size * 0.38, size * 0.43, 0, 0, Math.PI * 2);
      oCtx.stroke();

      // Brain parenchyma (soft tissue gray)
      oCtx.fillStyle = 'rgba(95, 105, 120, 0.4)';
      oCtx.beginPath();
      oCtx.ellipse(size * 0.5, size * 0.5, size * 0.35, size * 0.40, 0, 0, Math.PI * 2);
      oCtx.fill();

      // Interhemispheric fissure
      oCtx.strokeStyle = 'rgba(0, 0, 0, 0.45)';
      oCtx.lineWidth = 1.5;
      oCtx.beginPath();
      oCtx.moveTo(size * 0.5, size * 0.16);
      oCtx.lineTo(size * 0.5, size * 0.84);
      oCtx.stroke();

      // Ventricles with mild contralateral mass effect displacement
      oCtx.fillStyle = '#060912';
      oCtx.beginPath();
      oCtx.ellipse(size * 0.44, size * 0.48, size * 0.02, size * 0.07, -0.05, 0, Math.PI * 2);
      oCtx.ellipse(size * 0.53, size * 0.49, size * 0.015, size * 0.05, 0.08, 0, Math.PI * 2);
      oCtx.fill();

      // Acute Intracranial Hemorrhage / Brain Bleed (Hyperdense bright hematoma in basal ganglia / temporal lobe)
      oCtx.fillStyle = 'rgba(255, 255, 255, 0.95)';
      oCtx.beginPath();
      oCtx.ellipse(size * 0.63, size * 0.45, size * 0.08, size * 0.10, 0.25, 0, Math.PI * 2);
      oCtx.fill();

      // Surrounding hypodense peri-hematomal edema ring
      oCtx.strokeStyle = 'rgba(30, 35, 45, 0.7)';
      oCtx.lineWidth = 3;
      oCtx.stroke();

      // Secondary focal hyperdensity (satellite hemorrhage)
      oCtx.fillStyle = 'rgba(245, 245, 255, 0.85)';
      oCtx.beginPath();
      oCtx.arc(size * 0.58, size * 0.56, size * 0.035, 0, Math.PI * 2);
      oCtx.fill();

      // Grad-CAM attention focused directly on acute brain bleed
      const grad = gCtx.createRadialGradient(size * 0.63, size * 0.45, 4, size * 0.63, size * 0.45, size * 0.24);
      grad.addColorStop(0, 'rgba(255, 10, 20, 0.95)');
      grad.addColorStop(0.35, 'rgba(255, 140, 0, 0.85)');
      grad.addColorStop(0.7, 'rgba(0, 210, 255, 0.5)');
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      gCtx.fillStyle = grad;
      gCtx.fillRect(0, 0, size, size);

    } else if (subtype === 'normal') {
      // 2D CT Thoracic slice: Normal lung fields
      drawThoracicCt(oCtx, size, false);
      const grad = gCtx.createRadialGradient(size * 0.5, size * 0.5, 10, size * 0.5, size * 0.5, size * 0.35);
      grad.addColorStop(0, 'rgba(0, 180, 255, 0.35)');
      grad.addColorStop(1, 'rgba(0, 0, 40, 0)');
      gCtx.fillStyle = grad;
      gCtx.fillRect(0, 0, size, size);

    } else {
      // 2D CT Thoracic slice with COVID-19 / Pneumonia opacities
      drawThoracicCt(oCtx, size, true);
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
    }

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

    if (subtype === 'dementia') {
      // Dementia: Enlarged lateral ventricles (ventriculomegaly) + cortical sulcal widening
      oCtx.fillStyle = 'rgba(12, 18, 30, 0.95)';
      oCtx.beginPath();
      oCtx.ellipse(size * 0.43, size * 0.48, size * 0.055, size * 0.12, -0.15, 0, Math.PI * 2);
      oCtx.ellipse(size * 0.57, size * 0.48, size * 0.055, size * 0.12, 0.15, 0, Math.PI * 2);
      oCtx.fill();

      // Third ventricle dilation
      oCtx.fillRect(size * 0.485, size * 0.44, size * 0.03, size * 0.10);

      // Prominent widened sulci along cortex (cerebral atrophy pattern)
      oCtx.strokeStyle = 'rgba(10, 15, 25, 0.85)';
      oCtx.lineWidth = 2.5;
      for (let angle = 0; angle < Math.PI * 2; angle += 0.45) {
        if (Math.abs(Math.sin(angle)) > 0.2) {
          const x1 = size * 0.5 + Math.cos(angle) * (size * 0.34);
          const y1 = size * 0.5 + Math.sin(angle) * (size * 0.40);
          const x2 = size * 0.5 + Math.cos(angle) * (size * 0.28);
          const y2 = size * 0.5 + Math.sin(angle) * (size * 0.34);
          oCtx.beginPath();
          oCtx.moveTo(x1, y1);
          oCtx.lineTo(x2, y2);
          oCtx.stroke();
        }
      }

      // Grad-CAM attention highlighting ventricular dilation and temporal atrophy
      const gradV1 = gCtx.createRadialGradient(size * 0.43, size * 0.48, 5, size * 0.43, size * 0.48, size * 0.18);
      gradV1.addColorStop(0, 'rgba(255, 30, 30, 0.95)');
      gradV1.addColorStop(0.4, 'rgba(255, 170, 0, 0.8)');
      gradV1.addColorStop(0.7, 'rgba(0, 200, 255, 0.45)');
      gradV1.addColorStop(1, 'rgba(0, 0, 0, 0)');
      gCtx.fillStyle = gradV1;
      gCtx.fillRect(0, 0, size, size);

      const gradV2 = gCtx.createRadialGradient(size * 0.57, size * 0.48, 5, size * 0.57, size * 0.48, size * 0.18);
      gradV2.addColorStop(0, 'rgba(255, 30, 30, 0.95)');
      gradV2.addColorStop(0.4, 'rgba(255, 170, 0, 0.8)');
      gradV2.addColorStop(0.7, 'rgba(0, 200, 255, 0.45)');
      gradV2.addColorStop(1, 'rgba(0, 0, 0, 0)');
      gCtx.fillStyle = gradV2;
      gCtx.fillRect(0, 0, size, size);

    } else if (subtype === 'notumor') {
      // Normal Brain: normal small ventricles
      oCtx.fillStyle = 'rgba(20, 30, 50, 0.8)';
      oCtx.beginPath();
      oCtx.ellipse(size * 0.46, size * 0.48, size * 0.03, size * 0.08, -0.1, 0, Math.PI * 2);
      oCtx.ellipse(size * 0.54, size * 0.48, size * 0.03, size * 0.08, 0.1, 0, Math.PI * 2);
      oCtx.fill();

      // Normal Brain: low uniform response
      const grad = gCtx.createRadialGradient(size * 0.5, size * 0.5, 10, size * 0.5, size * 0.5, size * 0.3);
      grad.addColorStop(0, 'rgba(0, 180, 255, 0.35)');
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      gCtx.fillStyle = grad;
      gCtx.fillRect(0, 0, size, size);

    } else {
      // Standard Brain Tumor: normal ventricles + hyperintense tumor lesion
      oCtx.fillStyle = 'rgba(20, 30, 50, 0.8)';
      oCtx.beginPath();
      oCtx.ellipse(size * 0.46, size * 0.48, size * 0.03, size * 0.08, -0.1, 0, Math.PI * 2);
      oCtx.ellipse(size * 0.54, size * 0.48, size * 0.03, size * 0.08, 0.1, 0, Math.PI * 2);
      oCtx.fill();

      // Hyperintense Tumor mass in left frontal-parietal region
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
    }
  }

  // Generate blended overlay canvas
  const overlayCanvas = document.createElement('canvas');
  overlayCanvas.width = size;
  overlayCanvas.height = size;
  const ovCtx = overlayCanvas.getContext('2d');
  ovCtx.drawImage(originalCanvas, 0, 0);
  ovCtx.globalAlpha = 0.55;
  ovCtx.drawImage(gradcamCanvas, 0, 0);
  ovCtx.globalAlpha = 1.0;

  const originalUrl = originalCanvas.toDataURL('image/png');
  const heatmapUrl = gradcamCanvas.toDataURL('image/png');
  const overlayUrl = overlayCanvas.toDataURL('image/png');

  const visualizations = {
    original: originalUrl,
    overlay: overlayUrl,
    heatmap: heatmapUrl,
  };

  if (modality === 'mri') {
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = size;
    cropCanvas.height = size;
    const cCtx = cropCanvas.getContext('2d');
    cCtx.fillStyle = '#060a12';
    cCtx.fillRect(0, 0, size, size);
    cCtx.drawImage(originalCanvas, size * 0.1, size * 0.08, size * 0.8, size * 0.84, size * 0.05, size * 0.05, size * 0.9, size * 0.9);
    visualizations.cropped = cropCanvas.toDataURL('image/png');
  }

  return {
    original: originalUrl,
    gradcam: overlayUrl,
    visualizations,
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

function drawThoracicCt(ctx, size, withOpacities) {
  // Body contour
  ctx.fillStyle = 'rgba(100, 115, 140, 0.25)';
  ctx.beginPath();
  ctx.ellipse(size * 0.5, size * 0.5, size * 0.42, size * 0.36, 0, 0, Math.PI * 2);
  ctx.fill();

  // Thoracic wall / ribs
  ctx.strokeStyle = 'rgba(240, 245, 255, 0.85)';
  ctx.lineWidth = 4;
  ctx.stroke();

  // Spine at posterior
  ctx.fillStyle = 'rgba(250, 250, 255, 0.95)';
  ctx.beginPath();
  ctx.arc(size * 0.5, size * 0.72, size * 0.08, 0, Math.PI * 2);
  ctx.fill();

  // Lung fields (hypodense / dark air)
  ctx.fillStyle = '#070b14';
  ctx.beginPath();
  ctx.ellipse(size * 0.35, size * 0.48, size * 0.12, size * 0.16, -0.15, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(size * 0.65, size * 0.48, size * 0.12, size * 0.16, 0.15, 0, Math.PI * 2);
  ctx.fill();

  if (withOpacities) {
    // Ground glass opacities / infiltrates (subtle peripheral haziness)
    ctx.fillStyle = 'rgba(200, 220, 240, 0.55)';
    ctx.beginPath();
    ctx.arc(size * 0.72, size * 0.45, size * 0.06, 0, Math.PI * 2);
    ctx.arc(size * 0.30, size * 0.52, size * 0.05, 0, Math.PI * 2);
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
    name: 'DenseNet-121 (Thoracic Slice Classifier)',
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
  ct_brain_bleed: {
    name: 'CQ500 ResNet-50 (Cranial Hemorrhage Classifier)',
    framework: 'PyTorch 2.2',
    input_size: '512x512 px (Windowed Cranial CT Slice)',
    threshold: 0.70,
    dataset: 'CQ500 & RSNA Intracranial Hemorrhage Challenge (874,035 CT slices)',
    metrics: {
      'ROC-AUC': '0.974',
      'Sensitivity': '94.2%',
      'Specificity': '93.5%',
      'F1-Score': '0.938',
    },
    classes: ['NO BLEED', 'BRAIN BLEED (HEMORRHAGE)'],
  },
  mri: {
    name: 'ResNet-50 Classifier (Brain Tumor)',
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
  mri_dementia: {
    name: 'DenseNet-121 (Neurodegenerative Dementia Stager)',
    framework: 'PyTorch 2.2',
    input_size: '224x224 px (T1-weighted Axial/Coronal)',
    threshold: 0.65,
    dataset: 'ADNI & OASIS-1 Brain MRI Cohort (Alzheimer’s Disease Neuroimaging)',
    metrics: {
      'Accuracy': '92.6%',
      'ROC-AUC': '0.958',
      'Sensitivity': '91.3%',
      'Specificity': '93.7%',
    },
    classes: ['NON-DEMENTED', 'VERY MILD DEMENTIA', 'MILD DEMENTIA', 'MODERATE DEMENTIA'],
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
    label: 'Thoracic CT Slice (COVID-19 Ground-Glass)',
    description: '2D Axial chest CT slice with peripheral bilateral opacities',
    subtype: 'default',
  },
  {
    id: 'ct_brain_bleed',
    modality: 'ct',
    label: 'Cranial CT Slice (Acute Brain Bleed)',
    description: 'Non-contrast head CT with acute hyperdense intraparenchymal hemorrhage',
    subtype: 'brain_bleed',
  },
  {
    id: 'ct_normal',
    modality: 'ct',
    label: 'Thoracic CT Slice (Normal Lung)',
    description: 'Clear lung parenchyma without consolidation or effusion',
    subtype: 'normal',
  },
  {
    id: 'mri_tumor',
    modality: 'mri',
    label: 'Brain MRI (Tumor Mass)',
    description: 'Axial T1ce/FLAIR brain slice with hyperintense lesion',
    subtype: 'default',
  },
  {
    id: 'mri_dementia',
    modality: 'mri',
    label: 'Brain MRI (Dementia / Cortical Atrophy)',
    description: 'Axial T1 MRI displaying marked ventricular enlargement and cortical atrophy',
    subtype: 'dementia',
  },
  {
    id: 'mri_notumor',
    modality: 'mri',
    label: 'Brain MRI (No Tumor)',
    description: 'Normal brain axial slice, symmetric ventricular system',
    subtype: 'notumor',
  },
];
