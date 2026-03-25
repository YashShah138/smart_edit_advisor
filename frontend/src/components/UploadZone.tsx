import { useState, useCallback, useRef, useEffect } from 'react'

const ACCEPTED_EXTENSIONS = ['.cr2', '.nef', '.arw', '.dng', '.jpg', '.jpeg', '.png']
const MAX_SIZE_MB = 50
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

interface Props {
  onFileSelect: (file: File) => void
  selectedFile: File | null
  disabled?: boolean
}

const PREVIEWABLE = ['.jpg', '.jpeg', '.png']

export default function UploadZone({ onFileSelect, selectedFile, disabled }: Props) {
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [previewDims, setPreviewDims] = useState<{ w: number; h: number } | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Generate a preview URL for browser-renderable formats
  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null)
      setPreviewDims(null)
      return
    }
    const ext = '.' + selectedFile.name.split('.').pop()?.toLowerCase()
    if (!PREVIEWABLE.includes(ext)) {
      setPreviewUrl(null)
      setPreviewDims(null)
      return
    }
    const url = URL.createObjectURL(selectedFile)
    const img = new Image()
    img.onload = () => setPreviewDims({ w: img.naturalWidth, h: img.naturalHeight })
    img.src = url
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [selectedFile])

  const validate = useCallback((file: File): string | null => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      return `Unsupported format "${ext}". Accepted: ${ACCEPTED_EXTENSIONS.join(', ')}`
    }
    if (file.size > MAX_SIZE_BYTES) {
      return `File too large (${(file.size / 1e6).toFixed(1)}MB). Maximum: ${MAX_SIZE_MB}MB`
    }
    return null
  }, [])

  const handleFile = useCallback((file: File) => {
    const err = validate(file)
    if (err) { setError(err); return }
    setError(null)
    onFileSelect(file)
  }, [validate, onFileSelect])

  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); if (!disabled) setIsDragging(true) }
  const onDragLeave = () => setIsDragging(false)
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragging(false)
    if (disabled) return
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }
  const onClick = () => { if (!disabled) inputRef.current?.click() }
  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
    e.target.value = ''
  }

  return (
    <div className="w-full">
      <div
        onClick={onClick}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`
          relative rounded-xl border-2 border-dashed transition-all duration-200 cursor-pointer
          ${isDragging
            ? 'border-accent bg-orange-50/60 scale-[1.01]'
            : selectedFile
              ? 'border-green-500/60 bg-green-50/40'
              : 'border-surface-400 border-stone-400 bg-surface-100/40'
          }
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          px-6 py-10 text-center
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(',')}
          onChange={onInputChange}
          className="hidden"
        />

        {selectedFile ? (
          <div className="space-y-3">
            {previewUrl ? (
              <div className="mx-auto max-h-64 overflow-hidden rounded-lg">
                <img
                  src={previewUrl}
                  alt="Preview"
                  style={{ objectFit: 'contain', width: '100%', maxHeight: '16rem' }}
                />
              </div>
            ) : (
              <div className="w-12 h-12 mx-auto rounded-full bg-green-500/15 border border-green-500/30 flex items-center justify-center">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </div>
            )}
            <p className="text-sm font-medium text-stone-800">{selectedFile.name}</p>
            <p className="text-xs text-stone-500">
              {(selectedFile.size / 1e6).toFixed(1)} MB
              <span className="mx-2">·</span>
              {selectedFile.name.split('.').pop()?.toUpperCase()}
              {previewDims && (
                <>
                  <span className="mx-2">·</span>
                  {previewDims.w} × {previewDims.h}
                </>
              )}
            </p>
            <p className="text-xs text-stone-400">Click to change file</p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="w-14 h-14 mx-auto rounded-full bg-surface-300 border border-surface-400 flex items-center justify-center">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-stone-400">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <div>
              <p className="text-sm text-stone-700">
                <span className="text-accent font-medium">Click to upload</span> or drag and drop
              </p>
              <p className="text-xs text-stone-500 mt-1">RAW files: CR2, NEF, ARW, DNG &middot; Also accepts JPG, PNG</p>
              <p className="text-xs text-stone-500 mt-0.5">Maximum file size: {MAX_SIZE_MB}MB</p>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-3 px-4 py-2 rounded-lg bg-red-50 border border-red-200 text-sm text-red-600">
          {error}
        </div>
      )}
    </div>
  )
}
