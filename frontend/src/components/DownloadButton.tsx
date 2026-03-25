interface Props {
  imageData: string
  filename?: string
}

export default function DownloadButton({ imageData, filename = 'enhanced.jpg' }: Props) {
  const handleDownload = () => {
    const link = document.createElement('a')
    link.href = imageData
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <button
      onClick={handleDownload}
      className="
        flex items-center gap-2 px-5 py-2.5 rounded-lg
        bg-surface-300 hover:bg-surface-400 border border-surface-400 hover:border-stone-400
        text-sm font-medium text-stone-800
        transition-all duration-200 hover:scale-[1.02]
      "
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="7 10 12 15 17 10" />
        <line x1="12" y1="15" x2="12" y2="3" />
      </svg>
      Download JPEG
    </button>
  )
}
