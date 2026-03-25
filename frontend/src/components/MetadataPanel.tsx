import { RawMetadata, StageTime } from '../lib/api'

interface Props {
  metadata: RawMetadata
  stages: StageTime[]
  processingTime: number
  profile: string
}

export default function MetadataPanel({ metadata, stages, processingTime, profile }: Props) {
  return (
    <div className="w-full glass-panel p-5">
      <h3 className="text-sm font-medium text-stone-500 mb-4">Processing Details</h3>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-stone-400 mb-1">Format</p>
          <p className="text-sm font-mono text-stone-800">{metadata.file_format}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-stone-400 mb-1">Resolution</p>
          <p className="text-sm font-mono text-stone-800">{metadata.width} &times; {metadata.height}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-stone-400 mb-1">File Size</p>
          <p className="text-sm font-mono text-stone-800">{metadata.file_size_mb} MB</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-stone-400 mb-1">Profile</p>
          <p className="text-sm font-mono text-accent">{profile.replace(/_/g, ' ')}</p>
        </div>

        {metadata.camera_model && metadata.camera_model !== 'Unknown' && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-stone-400 mb-1">Camera</p>
            <p className="text-sm font-mono text-stone-800">{metadata.camera_model}</p>
          </div>
        )}
        {metadata.iso && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-stone-400 mb-1">ISO</p>
            <p className="text-sm font-mono text-stone-800">{metadata.iso}</p>
          </div>
        )}
        {metadata.aperture && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-stone-400 mb-1">Aperture</p>
            <p className="text-sm font-mono text-stone-800">{metadata.aperture}</p>
          </div>
        )}
        {metadata.shutter_speed && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-stone-400 mb-1">Shutter</p>
            <p className="text-sm font-mono text-stone-800">{metadata.shutter_speed}</p>
          </div>
        )}
      </div>

      {/* Pipeline timing */}
      <div className="border-t border-surface-400 pt-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs font-medium text-stone-500 uppercase tracking-wider">Pipeline Timing</h4>
          <span className="text-xs font-mono text-accent">{(processingTime / 1000).toFixed(2)}s total</span>
        </div>

        <div className="space-y-2">
          {stages.map((s, i) => {
            const maxDuration = Math.max(...stages.map((st) => st.duration_ms), 1)
            const pct = (s.duration_ms / maxDuration) * 100

            return (
              <div key={i} className="flex items-center gap-3">
                <span className="text-xs text-stone-500 w-28 shrink-0">{s.name}</span>
                <div className="flex-1 h-1.5 rounded-full bg-surface-300 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-accent/70 to-accent"
                    style={{ width: `${Math.max(pct, 2)}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-stone-500 w-16 text-right">
                  {s.duration_ms < 1000 ? `${s.duration_ms.toFixed(0)}ms` : `${(s.duration_ms / 1000).toFixed(2)}s`}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
