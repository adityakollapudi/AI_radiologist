/**
 * Utility to identify whether an AI prediction represents a positive pathological
 * finding (e.g. Tumor, Pneumonia, Hemorrhage/Bleed, COVID-19, Dementia, Abnormal)
 * or a negative/normal/benign scan.
 */
export function isDiseasePositive(prediction) {
  if (!prediction || typeof prediction !== 'string') return false;
  const p = prediction.toUpperCase();

  // Explicit negative / healthy conditions
  if (
    p.includes('NO TUMOR') ||
    p.includes('NO BLEED') ||
    p.includes('NO HEMORRHAGE') ||
    p.includes('NON-DEMENTED') ||
    p.includes('BRAIN IS SAFE') ||
    p.includes('SAFE') ||
    (p.includes('NORMAL') && !p.includes('ABNORMAL')) ||
    p.includes('HEALTHY') ||
    p.includes('BENIGN')
  ) {
    return false;
  }

  // Explicit positive disease conditions or alerts
  if (
    p.includes('TUMOR') ||
    p.includes('HEMORRHAGE') ||
    p.includes('BLEED') ||
    p.includes('PNEUMONIA') ||
    p.includes('COVID') ||
    p.includes('ABNORMAL') ||
    p.includes('DEMENTIA') ||
    p.includes('ALERT') ||
    p.includes('CRITICAL') ||
    p.includes('POSITIVE')
  ) {
    return true;
  }

  return false;
}
