import { useState, useCallback } from 'react'
import Header from './components/Header'
import UploadZone from './components/UploadZone'
import ProfileSelector from './components/ProfileSelector'
import ProgressTracker from './components/ProgressTracker'
import BeforeAfter from './components/BeforeAfter'
import MetadataPanel from './components/MetadataPanel'
import DownloadButton from './components/DownloadButton'
import { useEnhance } from './hooks/useEnhance'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [profile, setProfile] = useState('expert_natural')
  const { stage, uploadProgress, result, error, enhance, reset } = useEnhance()

  const isProcessing = stage !== 'idle' && stage !== 'complete' && stage !== 'error'
  const hasResult = stage === 'complete' && result !== null

  const handleEnhance = useCallback(() => {
    if (!file) return
    enhance(file, profile)
  }, [file, profile, enhance])

  const handleReset = useCallback(() => {
    setFile(null)
    reset()
  }, [reset])

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <main className="flex-1 px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-6">

          {/* Hero text */}
          {!hasResult && (
            <div className="text-center mb-2">
              <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-stone-900 mb-2">
                Transform your RAW photos with AI
              </h2>
              <p className="text-stone-500 text-sm max-w-lg mx-auto">
                Upload a RAW file and the three-stage ML pipeline will denoise, sharpen, and color grade it
                to produce a professionally edited result.
              </p>
            </div>
          )}

          {/* Upload zone */}
          {!hasResult && (
            <UploadZone
              onFileSelect={setFile}
              selectedFile={file}
              disabled={isProcessing}
            />
          )}

          {/* Profile selector */}
          {!hasResult && (
            <ProfileSelector
              selected={profile}
              onSelect={setProfile}
              disabled={isProcessing}
            />
          )}

          {/* Enhance button */}
          {!hasResult && !isProcessing && file && (
            <div className="flex justify-center">
              <button
                onClick={handleEnhance}
                className="
                  px-8 py-3 rounded-xl font-semibold text-sm
                  bg-gradient-to-r from-accent to-accent-dark
                  hover:from-accent-light hover:to-accent
                  text-white shadow-lg shadow-accent/20
                  transition-all duration-200 hover:scale-[1.03]
                  active:scale-[0.98]
                "
              >
                Enhance Photo
              </button>
            </div>
          )}

          {/* Progress tracker */}
          {isProcessing && (
            <ProgressTracker stage={stage} uploadProgress={uploadProgress} />
          )}

          {/* Error display */}
          {stage === 'error' && error && (
            <div className="glass-panel p-5 border-red-300">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-red-100 border border-red-200 flex items-center justify-center shrink-0">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-medium text-red-700">Enhancement failed</p>
                  <p className="text-xs text-stone-500 mt-1">{error}</p>
                </div>
              </div>
              <button
                onClick={handleReset}
                className="mt-3 text-xs text-accent hover:text-accent-light transition-colors"
              >
                Try again
              </button>
            </div>
          )}

          {/* Results */}
          {hasResult && result && (
            <div className="space-y-6">
              <BeforeAfter before={result.before} after={result.result} />

              <div className="flex items-center justify-between">
                <button
                  onClick={handleReset}
                  className="
                    flex items-center gap-2 px-4 py-2 rounded-lg
                    text-sm text-stone-600 hover:text-stone-900
                    bg-surface-300 hover:bg-surface-400 border border-surface-400
                    transition-all duration-200
                  "
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="1 4 1 10 7 10" />
                    <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                  </svg>
                  New Photo
                </button>

                <DownloadButton
                  imageData={result.result}
                  filename={file ? file.name.replace(/\.\w+$/, '_enhanced.jpg') : 'enhanced.jpg'}
                />
              </div>

              <MetadataPanel
                metadata={result.metadata}
                stages={result.stages}
                processingTime={result.processing_time}
                profile={result.profile}
              />
            </div>
          )}

          {/* How it works */}
          {!hasResult && !isProcessing && (
            <div className="mt-12 border-t border-surface-400 pt-8">
              <h3 className="text-center text-sm font-medium text-stone-400 mb-6 uppercase tracking-wider">How it works</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                  {
                    step: '01',
                    title: 'Denoise',
                    desc: 'Edge-preserving noise reduction removes sensor noise while maintaining fine detail and texture.',
                  },
                  {
                    step: '02',
                    title: 'Sharpen',
                    desc: 'Multi-pass unsharp masking with clarity enhancement recovers micro-detail and adds structure.',
                  },
                  {
                    step: '03',
                    title: 'Color Grade',
                    desc: 'Parametric curve adjustments apply professional color profiles with split-toning and tone mapping.',
                  },
                ].map((item) => (
                  <div key={item.step} className="glass-panel p-5">
                    <span className="text-accent font-mono text-xs font-semibold">{item.step}</span>
                    <h4 className="text-sm font-semibold text-stone-900 mt-2 mb-1">{item.title}</h4>
                    <p className="text-xs text-stone-500 leading-relaxed">{item.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </main>

      <footer className="border-t border-surface-400 px-6 py-4 text-center bg-surface-100/50">
        <p className="text-xs text-stone-400">
          Built with FastAPI, OpenCV, and React &middot; Three-stage ML pipeline &middot; Supports CR2, NEF, ARW, DNG
        </p>
      </footer>
    </div>
  )
}
