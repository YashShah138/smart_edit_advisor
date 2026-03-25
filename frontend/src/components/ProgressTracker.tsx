import { ProcessingStage } from '../hooks/useEnhance'

const STAGES = [
  { key: 'decoding',      label: 'Decoding RAW', icon: '01' },
  { key: 'denoising',     label: 'Denoising',    icon: '02' },
  { key: 'sharpening',    label: 'Sharpening',   icon: '03' },
  { key: 'color_grading', label: 'Color Grading', icon: '04' },
] as const

const STAGE_ORDER = ['uploading', 'decoding', 'denoising', 'sharpening', 'color_grading', 'complete']

function getStageIndex(stage: ProcessingStage): number {
  return STAGE_ORDER.indexOf(stage)
}

interface Props {
  stage: ProcessingStage
  uploadProgress: number
}

export default function ProgressTracker({ stage, uploadProgress }: Props) {
  const currentIndex = getStageIndex(stage)

  return (
    <div className="w-full glass-panel p-5">
      {/* Upload progress */}
      {stage === 'uploading' && (
        <div className="mb-4">
          <div className="flex justify-between text-xs text-stone-500 mb-1.5">
            <span>Uploading...</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="h-1.5 rounded-full bg-surface-300 overflow-hidden">
            <div
              className="h-full bg-accent rounded-full transition-all duration-300 ease-out"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Pipeline stages */}
      <div className="flex items-center gap-2">
        {STAGES.map((s, i) => {
          const stageIdx = getStageIndex(s.key as ProcessingStage)
          const isActive   = s.key === stage
          const isComplete = currentIndex > stageIdx
          const isPending  = currentIndex < stageIdx

          return (
            <div key={s.key} className="flex items-center gap-2 flex-1">
              <div className="flex items-center gap-2 flex-1">
                {/* Step indicator */}
                <div
                  className={`
                    w-8 h-8 rounded-lg flex items-center justify-center text-xs font-mono font-medium shrink-0
                    transition-all duration-300
                    ${isComplete ? 'bg-green-100 text-green-700 border border-green-200' : ''}
                    ${isActive   ? 'bg-orange-100 text-accent border border-accent/30 animate-pulse' : ''}
                    ${isPending  ? 'bg-surface-300 text-stone-400 border border-surface-400' : ''}
                  `}
                >
                  {isComplete ? (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  ) : (
                    s.icon
                  )}
                </div>

                {/* Label */}
                <span className={`text-xs whitespace-nowrap ${
                  isActive   ? 'text-accent font-medium' :
                  isComplete ? 'text-green-700' :
                               'text-stone-400'
                }`}>
                  {s.label}
                </span>
              </div>

              {/* Connector line */}
              {i < STAGES.length - 1 && (
                <div className={`h-px flex-1 min-w-4 ${isComplete ? 'bg-green-300' : 'bg-surface-400'}`} />
              )}
            </div>
          )
        })}
      </div>

      {/* Status text */}
      <div className="mt-3 text-center">
        {stage === 'uploading'     && <p className="text-xs text-stone-500">Uploading file to server...</p>}
        {stage === 'decoding'      && <p className="text-xs text-accent">Demosaicing RAW sensor data...</p>}
        {stage === 'denoising'     && <p className="text-xs text-accent">Applying noise reduction...</p>}
        {stage === 'sharpening'    && <p className="text-xs text-accent">Enhancing detail &amp; micro-contrast...</p>}
        {stage === 'color_grading' && <p className="text-xs text-accent">Applying color profile...</p>}
      </div>
    </div>
  )
}
