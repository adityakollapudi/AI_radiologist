/**
 * Session storage & local persistence for analysis history & active session
 */

const STORAGE_KEY = 'medvision_last_analysis';
const PREF_KEY = 'medvision_preferences';

export function saveLastAnalysis(result) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(result));
  } catch (e) {
    console.warn('Unable to persist analysis result to sessionStorage', e);
  }
}

export function getLastAnalysis() {
  try {
    const data = sessionStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : null;
  } catch (e) {
    return null;
  }
}

export function clearLastAnalysis() {
  try {
    sessionStorage.removeItem(STORAGE_KEY);
  } catch (e) {
    // Ignore
  }
}

export function getPreferences() {
  try {
    const prefs = localStorage.getItem(PREF_KEY);
    return prefs ? JSON.parse(prefs) : { highContrast: false, reducedMotion: false };
  } catch (e) {
    return { highContrast: false, reducedMotion: false };
  }
}

export function savePreferences(prefs) {
  try {
    localStorage.setItem(PREF_KEY, JSON.stringify(prefs));
  } catch (e) {
    // Ignore
  }
}
