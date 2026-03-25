interface Profile {
  id: string
  name: string
  description: string
  aesthetic: string
}

const PROFILES: Profile[] = [
  {
    id: 'expert_natural',
    name: 'Expert Natural',
    description: 'Clean, neutral edit — professional hand-retouched look',
    aesthetic: 'Balanced · Natural · Clean',
  },
  {
    id: 'warm_film',
    name: 'Warm Film',
    description: 'Lifted shadows, warm midtones, slight highlight fade',
    aesthetic: 'Nostalgic · Golden · Soft',
  },
  {
    id: 'moody_contrast',
    name: 'Moody Contrast',
    description: 'Deep blacks, punchy midtones, desaturated highlights',
    aesthetic: 'Dramatic · Cinematic · Cool',
  },
  {
    id: 'bw_fine_art',
    name: 'B&W Fine Art',
    description: 'Luminosity-based black & white, high local contrast',
    aesthetic: 'Monochrome · Gallery · Tonal',
  },
  {
    id: 'golden_hour',
    name: 'Golden Hour',
    description: 'Warm orange grade, glowing highlights, rich shadows',
    aesthetic: 'Sunset · Warm · Romantic',
  },
  {
    id: 'clean_commercial',
    name: 'Clean Commercial',
    description: 'Bright, neutral, high clarity — product & portrait',
    aesthetic: 'Studio · Crisp · Vibrant',
  },
]

// Subtle tinted backgrounds for each card on the light theme
const TINTS: Record<string, string> = {
  expert_natural:   'from-stone-200/60 to-stone-100/40',
  warm_film:        'from-amber-100/70 to-orange-50/50',
  moody_contrast:   'from-slate-200/60 to-blue-100/30',
  bw_fine_art:      'from-stone-200/60 to-stone-100/40',
  golden_hour:      'from-orange-100/70 to-yellow-50/50',
  clean_commercial: 'from-slate-200/60 to-blue-100/30',
}

interface Props {
  selected: string
  onSelect: (id: string) => void
  disabled?: boolean
}

export default function ProfileSelector({ selected, onSelect, disabled }: Props) {
  return (
    <div className="w-full">
      <h3 className="text-sm font-medium text-stone-500 mb-3">Enhancement Profile</h3>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {PROFILES.map((p) => (
          <button
            key={p.id}
            onClick={() => !disabled && onSelect(p.id)}
            disabled={disabled}
            className={`
              relative text-left rounded-xl p-4 transition-all duration-200
              bg-gradient-to-br ${TINTS[p.id] || 'from-stone-100/60 to-stone-50/40'}
              border
              ${selected === p.id
                ? 'border-accent ring-1 ring-accent/40 accent-glow'
                : 'border-surface-400 hover:border-stone-400'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            {selected === p.id && (
              <div className="absolute top-2 right-2 w-5 h-5 rounded-full bg-accent flex items-center justify-center">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </div>
            )}
            <h4 className="text-sm font-semibold text-stone-900 mb-1">{p.name}</h4>
            <p className="text-xs text-stone-500 leading-relaxed mb-2">{p.description}</p>
            <p className="text-[10px] text-stone-400 font-mono uppercase tracking-wider">{p.aesthetic}</p>
          </button>
        ))}
      </div>
    </div>
  )
}
