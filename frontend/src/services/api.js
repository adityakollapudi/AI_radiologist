import { generateMedicalCanvasImages, VERIFIED_MODELS } from '../utils/mockData';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

/**
 * Health check endpoint for FastAPI backend
 */
export async function checkBackendHealth(signal = null) {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);
    const effectiveSignal = signal || controller.signal;

    const res = await fetch(`${BASE_URL}/health`, { signal: effectiveSignal });
    clearTimeout(timeoutId);
    if (!res.ok) return { online: false, error: `Backend responded with HTTP ${res.status}` };
    const data = await res.json();
    return { online: true, ...data };
  } catch (err) {
    return { online: false, error: err.name === 'AbortError' ? 'Health check timed out' : err.message };
  }
}

/**
 * Retrieve verified models list from backend (or fallback to verified specs)
 */
export async function getModels(signal = null) {
  try {
    const res = await fetch(`${BASE_URL}/models`, { signal });
    if (res.ok) {
      return await res.json();
    }
  } catch (e) {
    console.warn('Could not fetch remote model specs, using verified registry:', e.message);
  }
  return VERIFIED_MODELS;
}

/**
 * Helper to convert a data URL or base64 string to a standard File/Blob
 */
async function dataUrlToFile(dataUrl, filename = 'scan.png') {
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type || 'image/png' });
}

/**
 * Run real deep learning model prediction on a medical scan
 * @param {string} modality - 'xray' | 'ct' | 'mri'
 * @param {File|Blob|string} fileOrDataUrl - The uploaded image file or data URL preview
 * @param {object} options - Options including subtype override and AbortSignal
 */
export async function predictScan(modality, fileOrDataUrl, options = {}) {
  const normalizedModality = modality.toLowerCase();
  if (!['xray', 'ct', 'mri'].includes(normalizedModality)) {
    throw new Error(`Unsupported modality '${modality}'. Must be 'xray', 'ct', or 'mri'.`);
  }

  let fileToSend;
  if (fileOrDataUrl instanceof File) {
    fileToSend = fileOrDataUrl;
  } else if (fileOrDataUrl instanceof Blob) {
    fileToSend = new File([fileOrDataUrl], 'scan.png', { type: fileOrDataUrl.type || 'image/png' });
  } else if (typeof fileOrDataUrl === 'string' && fileOrDataUrl.startsWith('data:')) {
    fileToSend = await dataUrlToFile(fileOrDataUrl, `${normalizedModality}_scan.png`);
  } else {
    // If a preset string identifier was passed without a data URL, generate canvas to send to backend
    const subtype = options.subtype || 'default';
    const canvasImages = generateMedicalCanvasImages(normalizedModality, subtype);
    fileToSend = await dataUrlToFile(canvasImages.original, `${normalizedModality}_preset.png`);
  }

  const formData = new FormData();
  formData.append('file', fileToSend);
  if (options.subtype && options.subtype !== 'default') {
    formData.append('subtype', options.subtype);
  }

  try {
    const res = await fetch(`${BASE_URL}/predict/${normalizedModality}`, {
      method: 'POST',
      body: formData,
      signal: options.signal,
      // Note: We intentionally DO NOT set Content-Type header so the browser
      // sets multipart/form-data with the correct boundary automatically.
    });

    if (res.ok) {
      const data = await res.json();
      return data;
    }

    // Try parsing backend structured error message
    let errorMessage = `Inference failed with status ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson.error) {
        errorMessage = errJson.error;
      }
    } catch (_) {
      // Use fallback status text if body is not JSON
    }

    throw new Error(errorMessage);

  } catch (err) {
    // If request was aborted by user cancellation
    if (err.name === 'AbortError') {
      throw new Error('Analysis request was cancelled.');
    }

    // If explicit mock fallback is requested in environment for non-backend demo mode:
    const allowMockFallback = import.meta.env.VITE_USE_MOCK_FALLBACK === 'true';
    if (allowMockFallback) {
      console.warn('Real backend call failed, falling back to mock generator (VITE_USE_MOCK_FALLBACK is true):', err.message);
      return generateMockPrediction(normalizedModality, fileOrDataUrl, options);
    }

    // Otherwise, surface real network / backend error honestly per non-negotiable constraints
    console.error(`Backend call to ${BASE_URL}/predict/${normalizedModality} failed:`, err);
    throw new Error(`Failed to connect to MedVision AI backend at ${BASE_URL}: ${err.message}`);
  }
}

/**
 * Realistic mock inference generator with actual canvas rendering (used only when VITE_USE_MOCK_FALLBACK=true)
 */
async function generateMockPrediction(modality, file, options = {}) {
  await new Promise((resolve) => setTimeout(resolve, 800));

  const subtype = options.subtype || (typeof file === 'string' && file.includes('mura') ? 'musculoskeletal' : 'default');
  const canvasImages = generateMedicalCanvasImages(modality, subtype);

  const DISCLAIMER_TEXT =
    'Research and educational use only. This system does not provide a clinical diagnosis and must not replace evaluation by a qualified healthcare professional.';

  if (modality === 'xray') {
    if (subtype === 'musculoskeletal') {
      const model = VERIFIED_MODELS.xray_mura;
      return {
        modality: 'xray',
        routed_to: 'musculoskeletal',
        prediction: 'ABNORMAL',
        probabilities: { NORMAL: 0.12, ABNORMAL: 0.88 },
        confidence: 0.88,
        threshold: model.threshold,
        model_name: model.name,
        framework: model.framework,
        input_size: model.input_size,
        dataset: model.dataset,
        metrics: model.metrics,
        gradcam_png_base64: canvasImages.gradcam,
        original_png_base64: canvasImages.original,
        visualizations: canvasImages.visualizations,
        disclaimer: DISCLAIMER_TEXT,
        warnings: [],
      };
    } else {
      const model = VERIFIED_MODELS.xray_chest;
      return {
        modality: 'xray',
        routed_to: 'chest',
        prediction: 'PNEUMONIA',
        probabilities: { NORMAL: 0.08, PNEUMONIA: 0.92 },
        confidence: 0.92,
        threshold: model.threshold,
        model_name: model.name,
        framework: model.framework,
        input_size: model.input_size,
        dataset: model.dataset,
        metrics: model.metrics,
        gradcam_png_base64: canvasImages.gradcam,
        original_png_base64: canvasImages.original,
        visualizations: canvasImages.visualizations,
        disclaimer: DISCLAIMER_TEXT,
        warnings: [],
      };
    }
  } else if (modality === 'ct') {
    const isBrainBleed = subtype === 'brain_bleed';
    if (isBrainBleed) {
      const model = VERIFIED_MODELS.ct_brain_bleed;
      return {
        modality: 'ct',
        routed_to: 'ct_brain_bleed',
        prediction: 'BRAIN BLEED (HEMORRHAGE)',
        probabilities: { 'NO BLEED': 0.04, 'BRAIN BLEED (HEMORRHAGE)': 0.96 },
        confidence: 0.96,
        threshold: model.threshold,
        model_name: model.name,
        framework: model.framework,
        input_size: model.input_size,
        dataset: model.dataset,
        metrics: model.metrics,
        gradcam_png_base64: canvasImages.gradcam,
        original_png_base64: canvasImages.original,
        visualizations: canvasImages.visualizations,
        disclaimer: DISCLAIMER_TEXT,
        warnings: ['CRITICAL FINDING: Acute intracranial hyperdensity detected.'],
      };
    } else {
      const model = VERIFIED_MODELS.ct;
      return {
        modality: 'ct',
        routed_to: 'ct_axial_slice',
        prediction: 'COVID-19',
        probabilities: { NORMAL: 0.04, PNEUMONIA: 0.14, 'COVID-19': 0.82 },
        confidence: 0.82,
        threshold: model.threshold,
        model_name: model.name,
        framework: model.framework,
        input_size: model.input_size,
        dataset: model.dataset,
        metrics: model.metrics,
        gradcam_png_base64: canvasImages.gradcam,
        original_png_base64: canvasImages.original,
        visualizations: canvasImages.visualizations,
        disclaimer: DISCLAIMER_TEXT,
        warnings: [],
      };
    }
  } else if (modality === 'mri') {
    const isDementia = subtype === 'dementia';
    if (isDementia) {
      const model = VERIFIED_MODELS.mri_dementia;
      return {
        modality: 'mri',
        routed_to: 'mri_dementia',
        prediction: 'MILD DEMENTIA',
        probabilities: { 'NON-DEMENTED': 0.05, 'VERY MILD DEMENTIA': 0.16, 'MILD DEMENTIA': 0.74, 'MODERATE DEMENTIA': 0.05 },
        confidence: 0.74,
        threshold: model.threshold,
        model_name: model.name,
        framework: model.framework,
        input_size: model.input_size,
        dataset: model.dataset,
        metrics: model.metrics,
        gradcam_png_base64: canvasImages.gradcam,
        original_png_base64: canvasImages.original,
        visualizations: canvasImages.visualizations,
        disclaimer: DISCLAIMER_TEXT,
        warnings: ['NEURODEGENERATIVE FINDING: Structural pattern matches significant ventricular enlargement.'],
      };
    } else {
      const model = VERIFIED_MODELS.mri;
      return {
        modality: 'mri',
        routed_to: 'brain_mri',
        prediction: 'TUMOR',
        probabilities: { 'NO TUMOR': 0.06, TUMOR: 0.94 },
        confidence: 0.94,
        threshold: model.threshold,
        model_name: model.name,
        framework: model.framework,
        input_size: model.input_size,
        dataset: model.dataset,
        metrics: model.metrics,
        gradcam_png_base64: canvasImages.gradcam,
        original_png_base64: canvasImages.original,
        visualizations: canvasImages.visualizations,
        disclaimer: DISCLAIMER_TEXT,
        warnings: [],
      };
    }
  }

  throw new Error(`Unsupported modality: ${modality}`);
}

/**
 * Request PDF medical report generation from the backend and trigger browser download
 * @param {object} result - The prediction result object containing modality, prediction, images, metadata
 */
export async function downloadPdfReport(result) {
  if (!result) {
    throw new Error('No analysis result provided for PDF report generation.');
  }

  const res = await fetch(`${BASE_URL}/report/pdf`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(result),
  });

  if (!res.ok) {
    let errorDetail = `PDF generation failed with HTTP ${res.status}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) errorDetail = errJson.detail;
      else if (errJson.error) errorDetail = errJson.error;
    } catch (_) {}
    throw new Error(errorDetail);
  }

  const blob = await res.blob();
  if (blob.size === 0) {
    throw new Error('Received an empty PDF report from the server.');
  }

  // Determine filename from Content-Disposition header if present
  let filename = `MEDVISION_AI_Report_${result.modality || 'scan'}_${Date.now()}.pdf`;
  const disposition = res.headers.get('Content-Disposition');
  if (disposition && disposition.includes('filename=')) {
    const match = disposition.match(/filename=["']?([^"';]+)["']?/);
    if (match && match[1]) {
      filename = match[1];
    }
  }

  const blobUrl = window.URL.createObjectURL(blob);
  const downloadAnchor = document.createElement('a');
  downloadAnchor.style.display = 'none';
  downloadAnchor.href = blobUrl;
  downloadAnchor.download = filename;
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();

  setTimeout(() => {
    window.URL.revokeObjectURL(blobUrl);
    downloadAnchor.remove();
  }, 1000);

  return { success: true, filename };
}

/**
 * Send a query to the AI Radiologist Assistant with optional scan context
 * @param {string} message - User question
 * @param {object} context - Active scan context (prediction, modality, confidence)
 * @param {Array} history - Previous message history [{role: 'user'|'assistant', content: string}]
 */
export async function queryAssistant(message, context = null, history = []) {
  try {
    const res = await fetch(`${BASE_URL}/assistant/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        context,
        history,
      }),
    });

    if (res.ok) {
      return await res.json();
    }

    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Assistant responded with status ${res.status}`);
  } catch (err) {
    console.error('AI Assistant API call error:', err);
    throw err;
  }
}
