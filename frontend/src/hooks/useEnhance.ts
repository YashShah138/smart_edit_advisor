import { useState, useCallback } from 'react'
import { enhanceImage, EnhancementResponse } from '../lib/api'

export type ProcessingStage =
  | 'idle'
  | 'uploading'
  | 'decoding'
  | 'denoising'
  | 'sharpening'
  | 'color_grading'
  | 'complete'
  | 'error'

export interface EnhanceState {
  stage: ProcessingStage
  uploadProgress: number
  result: EnhancementResponse | null
  error: string | null
}

export function useEnhance() {
  const [state, setState] = useState<EnhanceState>({
    stage: 'idle',
    uploadProgress: 0,
    result: null,
    error: null,
  })

  const enhance = useCallback(async (file: File, profile: string) => {
    setState({ stage: 'uploading', uploadProgress: 0, result: null, error: null })

    try {
      // Simulate stage progression during upload
      const onProgress = (pct: number) => {
        setState((s) => ({ ...s, uploadProgress: pct }))
        if (pct >= 100) {
          // Upload done, now processing on server
          setState((s) => ({ ...s, stage: 'decoding' }))

          // Simulate stage progression timers
          setTimeout(() => setState((s) => s.stage === 'decoding' ? { ...s, stage: 'denoising' } : s), 800)
          setTimeout(() => setState((s) => s.stage === 'denoising' ? { ...s, stage: 'sharpening' } : s), 2000)
          setTimeout(() => setState((s) => s.stage === 'sharpening' ? { ...s, stage: 'color_grading' } : s), 3500)
        }
      }

      const result = await enhanceImage(file, profile, onProgress)
      setState({ stage: 'complete', uploadProgress: 100, result, error: null })
    } catch (err: any) {
      const message = err.response?.data?.detail || err.message || 'Enhancement failed'
      setState({ stage: 'error', uploadProgress: 0, result: null, error: message })
    }
  }, [])

  const reset = useCallback(() => {
    setState({ stage: 'idle', uploadProgress: 0, result: null, error: null })
  }, [])

  return { ...state, enhance, reset }
}
