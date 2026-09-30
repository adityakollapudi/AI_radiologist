import React from 'react';
import Navbar from './components/layout/Navbar';
import Home from './pages/Home';
import ScanSelection from './pages/ScanSelection';
import Upload from './pages/Upload';
import Results from './pages/Results';
import { useAnalysis } from './hooks/useAnalysis';
import { generateMedicalCanvasImages } from './utils/mockData';

export default function App() {
  const {
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
  } = useAnalysis();

  const handleFileSelected = (fileOrPresetId, _label = null) => {
    if (typeof fileOrPresetId === 'string') {

      // Sample Preset selected: generate high quality preview scan immediately
      const isMura = fileOrPresetId.includes('mura');
      const isNormal = fileOrPresetId.includes('normal');
      const isNoTumor = fileOrPresetId.includes('notumor');
      const subtype = isMura ? 'musculoskeletal' : isNormal ? 'normal' : isNoTumor ? 'notumor' : 'default';

      const images = generateMedicalCanvasImages(selectedModality, subtype);
      setScanFile(fileOrPresetId, images.original);
    } else {
      setScanFile(fileOrPresetId);
    }
  };

  return (
    <div className="min-h-screen bg-[#030712] text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Minimal Top Navbar */}
      <Navbar currentStep={currentStep} onNavigate={navigateTo} />

      {/* Main Continuous Workspace Container */}
      <main className="flex-1 pt-[62px] flex flex-col items-center justify-center relative w-full overflow-hidden">
        {currentStep === 'home' && (
          <Home
            onStartAnalysis={startAnalysis}
            onSelectModality={selectModality}
          />
        )}

        {currentStep === 'select' && (
          <ScanSelection
            onSelectModality={selectModality}
            onBack={goHome}
          />
        )}

        {currentStep === 'upload' && (
          <Upload
            modality={selectedModality}
            previewUrl={previewUrl}
            currentFile={currentFile}
            isAnalyzing={isAnalyzing}
            analysisPhase={analysisPhase}
            error={error}
            onFileSelected={handleFileSelected}
            onAnalyze={executeAnalysis}
            onChangeScan={() => setScanFile(null, null)}
            onBack={() => navigateTo('select')}
          />
        )}

        {currentStep === 'results' && (
          <Results
            result={analysisResult}
            onAnalyzeAnother={resetAll}
            onBackToSelection={() => navigateTo('select')}
          />
        )}
      </main>
    </div>
  );
}
