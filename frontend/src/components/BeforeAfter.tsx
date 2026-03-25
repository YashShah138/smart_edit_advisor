import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider'
import { useEffect, useState } from 'react'

interface Props {
  before: string  // base64 data URI
  after: string   // base64 data URI
}

export default function BeforeAfter({ before, after }: Props) {
  const [aspectRatio, setAspectRatio] = useState<string>('16 / 9')

  // Read the natural dimensions from the "after" image and lock the slider to that ratio
  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      if (img.naturalWidth && img.naturalHeight) {
        setAspectRatio(`${img.naturalWidth} / ${img.naturalHeight}`)
      }
    }
    img.src = after
  }, [after])

  return (
    <div className="w-full glass-panel overflow-hidden">
      <div className="relative" style={{ aspectRatio }}>
        <ReactCompareSlider
          itemOne={
            <ReactCompareSliderImage
              src={before}
              alt="Original RAW render"
              style={{ objectFit: 'contain', width: '100%', height: '100%', background: '#0a0a18' }}
            />
          }
          itemTwo={
            <ReactCompareSliderImage
              src={after}
              alt="Enhanced output"
              style={{ objectFit: 'contain', width: '100%', height: '100%', background: '#0a0a18' }}
            />
          }
          style={{ width: '100%', height: '100%' }}
          position={50}
          changePositionOnHover={false}
        />

        {/* Labels */}
        <div className="absolute top-3 left-3 px-2 py-1 rounded bg-stone-800/75 text-xs font-medium text-stone-200 backdrop-blur-sm pointer-events-none">
          Before
        </div>
        <div className="absolute top-3 right-3 px-2 py-1 rounded bg-accent/85 text-xs font-medium text-white backdrop-blur-sm pointer-events-none">
          After
        </div>
      </div>
    </div>
  )
}
