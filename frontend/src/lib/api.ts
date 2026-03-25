import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || ''

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120_000, // 2 min for large RAW files
})

export interface StageTime {
  name: string
  duration_ms: number
}

export interface RawMetadata {
  camera_model: string | null
  iso: number | null
  aperture: string | null
  shutter_speed: string | null
  focal_length: string | null
  width: number
  height: number
  file_format: string
  file_size_mb: number
}

export interface EnhancementResponse {
  result: string        // base64 JPEG (after)
  before: string        // base64 JPEG (before)
  session_id: string
  profile: string
  processing_time: number
  stages: StageTime[]
  metadata: RawMetadata
}

export interface ProfileInfo {
  id: string
  name: string
  description: string
  aesthetic: string
}

export async function enhanceImage(
  file: File,
  profile: string,
  onUploadProgress?: (pct: number) => void,
): Promise<EnhancementResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('profile', profile)

  const { data } = await api.post<EnhancementResponse>('/enhance', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (e.total && onUploadProgress) {
        onUploadProgress(Math.round((e.loaded / e.total) * 100))
      }
    },
  })
  return data
}

export async function getProfiles(): Promise<ProfileInfo[]> {
  const { data } = await api.get<ProfileInfo[]>('/profiles')
  return data
}

export async function healthCheck(): Promise<{ status: string }> {
  const { data } = await api.get('/health')
  return data
}
