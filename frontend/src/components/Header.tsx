export default function Header() {
  return (
    <header className="border-b border-surface-400 bg-surface-100/70 backdrop-blur-sm px-6 py-4">
      <div className="max-w-6xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent-dark flex items-center justify-center">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-stone-900">RAW Enhance</h1>
            <p className="text-xs text-stone-500">AI-Powered Photo Processing</p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-stone-500">
          <span className="px-2 py-1 rounded bg-surface-300 font-mono border border-surface-400">ML Pipeline</span>
          <span className="px-2 py-1 rounded bg-surface-300 font-mono border border-surface-400">3-Stage</span>
        </div>
      </div>
    </header>
  )
}
