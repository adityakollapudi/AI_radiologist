import React, { useState, useRef, useEffect } from 'react';
import {
  MessageSquare,
  Bot,
  X,
  Send,
  Sparkles,
  ChevronDown,
  RotateCcw,
  Activity,
  ShieldAlert,
  Loader2,
  FileText,
  HelpCircle,
  ArrowRight
} from 'lucide-react';
import { queryAssistant } from '../../services/api';
import { isDiseasePositive } from '../../utils/diseaseStatus';

export default function AIAssistantWidget({ currentStep, selectedModality, analysisResult }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        'Hi there! I am your **MedVision AI Radiology Guide**.\n\n' +
        'I am here to explain your scans and results in **plain, simple English** — without any confusing medical jargon!\n\n' +
        '- What your scan result means in everyday words\n' +
        '- What the colored highlights on your picture show\n' +
        '- What simple next steps you can take with your doctor\n' +
        '- How to download and print your PDF report\n\n' +
        'How can I help you today?',
      source: 'clinical_engine',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [suggestedQueries, setSuggestedQueries] = useState([
    'What do the colors on the scan mean?',
    'Explain my scan result in simple words',
    'How do I download the PDF report?',
    'What is the difference between CT and MRI?'
  ]);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
      inputRef.current?.focus();
    }
  }, [isOpen, messages, isLoading]);

  // Update suggestions dynamically if scan result is available
  useEffect(() => {
    if (analysisResult?.prediction) {
      setSuggestedQueries([
        `What does ${analysisResult.prediction} mean in simple words?`,
        'What do the colored highlights show?',
        'What should my next steps be?',
        'How sure is the AI about this result?'
      ]);
    }
  }, [analysisResult]);

  const handleSendMessage = async (textToSend = null) => {
    const text = (textToSend || inputValue).trim();
    if (!text || isLoading) return;

    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputValue('');
    setIsLoading(true);

    // Format context from analysisResult if present
    const context = analysisResult ? {
      modality: analysisResult.modality || selectedModality,
      prediction: analysisResult.prediction,
      confidence: analysisResult.confidence,
      routed_to: analysisResult.routed_to,
      model_name: analysisResult.model_name,
      probabilities: analysisResult.probabilities,
    } : (selectedModality ? { modality: selectedModality } : null);

    // Format history for backend
    const historyPayload = newMessages.slice(-6).map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      const response = await queryAssistant(text, context, historyPayload);
      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: response.reply,
        source: response.source,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, assistantMessage]);
      if (response.suggested_queries?.length > 0) {
        setSuggestedQueries(response.suggested_queries);
      }
    } catch (err) {
      const errorMessage = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content:
          'I encountered a temporary connection issue reaching the reasoning server. ' +
          'Please verify the backend is active at `http://127.0.0.1:8000`.',
        source: 'clinical_engine',
        isError: true,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([
      {
        id: 'welcome-reset',
        role: 'assistant',
        content: 'Conversation cleared. Feel free to ask any question about your scan or radiology concepts!',
        source: 'clinical_engine',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Helper to format simple markdown elements (bolding, headings, bullets)
  const formatMarkdown = (content) => {
    return content.split('\n').map((line, idx) => {
      if (line.startsWith('### ')) {
        return (
          <h4 key={idx} className="text-cyan-300 font-bold text-xs uppercase tracking-wider mt-2 mb-1">
            {line.replace('### ', '')}
          </h4>
        );
      }
      if (line.startsWith('> ')) {
        return (
          <blockquote key={idx} className="border-l-2 border-cyan-500/50 pl-2.5 py-1 my-1.5 text-[11px] text-slate-300 bg-cyan-950/20 rounded-r">
            {line.replace('> ', '')}
          </blockquote>
        );
      }
      if (line.startsWith('- ')) {
        const itemText = line.replace('- ', '');
        return (
          <li key={idx} className="text-[11px] leading-relaxed text-slate-200 ml-3 list-disc">
            <span dangerouslySetInnerHTML={{ __html: renderInlineFormatting(itemText) }} />
          </li>
        );
      }
      if (/^\d+\.\s/.test(line)) {
        const itemText = line.replace(/^\d+\.\s/, '');
        return (
          <li key={idx} className="text-[11px] leading-relaxed text-slate-200 ml-4 list-decimal">
            <span dangerouslySetInnerHTML={{ __html: renderInlineFormatting(itemText) }} />
          </li>
        );
      }
      if (line.trim() === '') {
        return <div key={idx} className="h-1.5" />;
      }
      return (
        <p key={idx} className="text-[11px] leading-relaxed text-slate-200 my-0.5" dangerouslySetInnerHTML={{ __html: renderInlineFormatting(line) }} />
      );
    });
  };

  const renderInlineFormatting = (text) => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
      .replace(/\*(.*?)\*/g, '<em class="text-cyan-200">$1</em>')
      .replace(/`([^`]+)`/g, '<code class="bg-black/40 px-1 py-0.5 rounded text-cyan-300 font-mono text-[10px]">$1</code>');
  };

  return (
    <>
      {/* Floating Launcher Button */}
      <div className="fixed bottom-6 right-6 z-50 flex items-center gap-3">
        {!isOpen && analysisResult?.prediction && (
          <div
            onClick={() => setIsOpen(true)}
            className="hidden md:flex items-center gap-2 px-3.5 py-2 rounded-full bg-slate-900/90 border border-cyan-500/40 text-xs font-mono text-cyan-300 shadow-[0_4px_20px_rgba(6,182,212,0.25)] cursor-pointer backdrop-blur hover:bg-slate-800 transition-all animate-bounce"
          >
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span>Ask AI about {analysisResult.prediction}</span>
          </div>
        )}

        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          aria-label="Toggle AI Assistant"
          className="relative group p-3.5 sm:p-4 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold shadow-[0_0_25px_rgba(6,182,212,0.45)] hover:shadow-[0_0_35px_rgba(6,182,212,0.6)] transition-all cursor-pointer transform hover:scale-105 active:scale-95"
        >
          {isOpen ? (
            <X className="w-5 h-5 text-slate-950" />
          ) : (
            <div className="flex items-center gap-2">
              <Bot className="w-5 h-5 text-slate-950" />
              <span className="hidden sm:inline font-mono text-xs uppercase tracking-wider font-extrabold pr-0.5">
                AI Copilot
              </span>
            </div>
          )}
          {!isOpen && (
            <span className="absolute -top-1 -right-1 flex h-3.5 w-3.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-cyan-400 border-2 border-slate-950" />
            </span>
          )}
        </button>
      </div>

      {/* Expanded Chat HUD Window */}
      {isOpen && (
        <div className="fixed bottom-22 right-4 sm:right-6 z-50 w-[calc(100vw-2rem)] sm:w-[440px] h-[580px] max-h-[82vh] bg-[#070e1b]/95 border border-cyan-500/30 rounded-2xl shadow-[0_10px_50px_rgba(0,0,0,0.85),0_0_25px_rgba(6,182,212,0.2)] backdrop-blur-2xl flex flex-col overflow-hidden animate-in fade-in slide-in-from-bottom-5 duration-200">
          
          {/* Header Bar */}
          <div className="px-4 py-3 bg-slate-900/80 border-b border-white/10 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
                <Bot className="w-4 h-4" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="font-heading font-bold text-sm text-white tracking-wide">
                    MEDVISION COPILOT
                  </h3>
                  <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-cyan-950 border border-cyan-500/40 text-cyan-300 uppercase">
                    AI Radiologist
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-[10px] text-slate-400 font-mono">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span>Clinical Engine Active</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={handleClearChat}
                title="Clear conversation"
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
              >
                <ChevronDown className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Active Context Banner (if scan is loaded) */}
          {analysisResult && (
            <div className={`px-3.5 py-1.5 border-b flex items-center justify-between text-[10px] font-mono ${
              isDiseasePositive(analysisResult.prediction)
                ? 'bg-red-950/40 border-red-500/20 text-red-300'
                : 'bg-cyan-950/40 border-cyan-500/20 text-cyan-300'
            }`}>
              <div className="flex items-center gap-1.5 truncate">
                <Activity className={`w-3 h-3 shrink-0 ${
                  isDiseasePositive(analysisResult.prediction) ? 'text-red-400' : 'text-cyan-400'
                }`} />
                <span className="truncate">
                  {analysisResult.modality?.toUpperCase()}:{' '}
                  <strong className={isDiseasePositive(analysisResult.prediction) ? 'text-red-400 font-bold' : 'text-emerald-400 font-bold'}>
                    {analysisResult.prediction}
                  </strong>{' '}
                  ({analysisResult.confidence ? `${(analysisResult.confidence * 100).toFixed(1)}%` : 'Verified'})
                </span>
              </div>
              <span className={`text-[9px] uppercase font-bold shrink-0 ml-2 ${
                isDiseasePositive(analysisResult.prediction) ? 'text-red-400/90' : 'text-cyan-400/80'
              }`}>
                Context Linked
              </span>
            </div>
          )}

          {/* Message Thread */}
          <div className="flex-1 p-3.5 sm:p-4 overflow-y-auto space-y-3.5 font-sans">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`max-w-[88%] px-3.5 py-2.5 rounded-2xl text-xs ${
                    msg.role === 'user'
                      ? 'bg-cyan-950/60 border border-cyan-500/40 text-cyan-50 rounded-tr-sm shadow-[0_2px_12px_rgba(6,182,212,0.15)]'
                      : msg.isError
                      ? 'bg-rose-950/50 border border-rose-500/30 text-rose-200 rounded-tl-sm'
                      : 'bg-slate-900/90 border border-white/10 text-slate-200 rounded-tl-sm shadow-[0_2px_10px_rgba(0,0,0,0.4)]'
                  }`}
                >
                  {formatMarkdown(msg.content)}
                </div>

                <div className="flex items-center gap-2 mt-1 px-1 text-[9px] font-mono text-slate-500">
                  <span>{msg.timestamp}</span>
                  {msg.source && (
                    <span className="uppercase text-[8px] text-cyan-500/80">
                      • {msg.source === 'gemini' ? 'Gemini 1.5' : 'Clinical Engine'}
                    </span>
                  )}
                </div>
              </div>
            ))}

            {/* Thinking / Loading Animation */}
            {isLoading && (
              <div className="flex items-start gap-2">
                <div className="bg-slate-900/90 border border-cyan-500/30 rounded-2xl rounded-tl-sm px-4 py-2.5 flex items-center gap-2 text-xs font-mono text-cyan-300">
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                  <span>Synthesizing radiological reasoning...</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Suggested Quick Questions Chips */}
          <div className="px-3.5 py-2 bg-black/40 border-t border-white/5 flex items-center gap-1.5 overflow-x-auto no-scrollbar">
            {suggestedQueries.map((query, index) => (
              <button
                key={index}
                type="button"
                onClick={() => handleSendMessage(query)}
                disabled={isLoading}
                className="whitespace-nowrap px-2.5 py-1 rounded-full bg-white/5 hover:bg-cyan-500/20 border border-white/10 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-200 text-[10px] font-mono transition-all cursor-pointer disabled:opacity-40"
              >
                {query}
              </button>
            ))}
          </div>

          {/* Input Bar */}
          <div className="p-3 bg-slate-900/90 border-t border-white/10 flex items-center gap-2">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              placeholder={analysisResult ? `Ask about this ${analysisResult.prediction} scan...` : 'Ask a medical imaging question...'}
              className="flex-1 bg-black/50 border border-white/10 focus:border-cyan-500/50 rounded-xl px-3.5 py-2 text-xs font-mono text-slate-100 placeholder:text-slate-500 outline-none transition-colors"
            />
            <button
              type="button"
              onClick={() => handleSendMessage()}
              disabled={!inputValue.trim() || isLoading}
              className="p-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold transition-all disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>

          {/* Safe Disclaimer Footer */}
          <div className="px-3 py-1 bg-black/80 text-[8.5px] font-mono text-slate-500 text-center border-t border-white/5">
            AI research prototype • For educational and demo purposes only
          </div>
        </div>
      )}
    </>
  );
}
