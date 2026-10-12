import { useState, useCallback } from 'react';
import { predictScan } from '../services/api';
import { saveLastAnalysis, getLastAnalysis, clearLastAnalysis } from '../utils/storage';

export function useAnalysis() {
  const [currentStep, setCurrentStep] = useState('home'); // 'home' | 'select' | 'upload' | 'results'
  
  // Lazy initialize previous session if present on initial load
  const [analysisResult, setAnalysisResult] = useState(() => {
    const saved = getLastAnalysis();
    return saved && saved.prediction ? saved : null;
  });

  const [selectedModality, setSelectedModality] = useState(() => {
    const saved = getLastAnalysis();
    return saved && saved.modality ? saved.modality : null;
  });

  const [currentFile, setCurrentFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisPhase, setAnalysisPhase] = useState('idle'); // 'preparing' | 'running' | 'gradcam' | 'complete'
  const [error, setError] = useState(null);

  const navigateTo = useCallback((step) => {
    setError(null);
    setCurrentStep(step);
  }, []);

  const startAnalysis = useCallback(() => {
    navigateTo('select');
  }, [navigateTo]);

  const selectModality = useCallback((modality) => {
    setSelectedModality(modality);
    setCurrentFile(null);
    setPreviewUrl(null);
    navigateTo('upload');
  }, [navigateTo]);

  const setScanFile = useCallback((file, preview = null) => {
    setCurrentFile(file);
    if (preview) {
      setPreviewUrl(preview);
    } else if (file instanceof File) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else if (typeof file === 'string') {
      setPreviewUrl(file);
    }
  }, []);

  const executeAnalysis = useCallback(async (options = {}) => {
    if (!selectedModality) {
      setError('Please select a medical imaging modality first.');
      return;
    }
    if (!currentFile && !previewUrl) {
      setError('Please upload or select a medical scan.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      // Step 1: PREPARING IMAGE
      setAnalysisPhase('preparing');
      await new Promise((res) => setTimeout(res, 900));

      // Step 2: RUNNING AI MODEL
      setAnalysisPhase('running');
      await new Promise((res) => setTimeout(res, 1200));

      // Step 3: GENERATING GRAD-CAM
      setAnalysisPhase('gradcam');
      const result = await predictScan(selectedModality, currentFile || previewUrl, options);
      await new Promise((res) => setTimeout(res, 1000));

      // Step 4: ANALYSIS COMPLETE
      setAnalysisPhase('complete');
      await new Promise((res) => setTimeout(res, 500));

      setAnalysisResult(result);
      saveLastAnalysis(result);
      setIsAnalyzing(false);
      navigateTo('results');
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.message || 'Analysis failed. Please try again.');
      setIsAnalyzing(false);
      setAnalysisPhase('idle');
    }
  }, [selectedModality, currentFile, previewUrl, navigateTo]);

  const resetAll = useCallback(() => {
    clearLastAnalysis();
    setAnalysisResult(null);
    setCurrentFile(null);
    setPreviewUrl(null);
    setSelectedModality(null);
    setAnalysisPhase('idle');
    setIsAnalyzing(false);
    setError(null);
    navigateTo('select');
  }, [navigateTo]);

  const goHome = useCallback(() => {
    navigateTo('home');
  }, [navigateTo]);

  return {
    currentStep,
    selectedModality,
    currentFile,
    previewUrl,
    isAnalyzing,
    analysisPhase,
    analysisResult,
    error,
    navigateTo,
    startAnalysis,
    selectModality,
    setScanFile,
    executeAnalysis,
    resetAll,
    goHome,
  };
}
