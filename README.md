# AI RAW Photo Enhancement Pipeline

A web application that transforms RAW camera files into professionally-edited photos using a three-stage ML pipeline: **Denoise → Sharpen → Color Grade**.

## Tech Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS, Vite
- **Backend**: Python, FastAPI, OpenCV, rawpy
- **ML Pipeline**: DnCNN (denoise), Real-ESRGAN (sharpen), DPED CNN (color grade)
- **Processing**: Patch-based for large images, session caching for RAW decode

## Quick Start

### Backend

```bash
cd raw-enhance
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) — the frontend proxies API calls to port 8000.

## Supported Formats


| Format    | Extension | Type     |
| --------- | --------- | -------- |
| Canon RAW | .CR2      | RAW      |
| Nikon RAW | .NEF      | RAW      |
| Sony RAW  | .ARW      | RAW      |
| Adobe DNG | .DNG      | RAW      |
| JPEG      | .JPG      | Fallback |
| PNG       | .PNG      | Fallback |


## Enhancement Profiles


| Profile          | Aesthetic                           |
| ---------------- | ----------------------------------- |
| Expert Natural   | Clean, neutral, professional        |
| Warm Film        | Nostalgic, warm midtones, faded     |
| Moody Contrast   | Cinematic, deep blacks, desaturated |
| B&W Fine Art     | Gallery-grade monochrome            |
| Golden Hour      | Sunset warmth, golden glow          |
| Clean Commercial | Studio-bright, crisp, vibrant       |


## API Endpoints

```
POST /enhance        Upload RAW + profile → enhanced JPEG
GET  /profiles       List available profiles
GET  /health         Health check
```

## ML Pipeline Architecture

```
RAW File → rawpy decode → float32 RGB [0,1]
  → Stage 1: Denoise (bilateral filter / DnCNN)
  → Stage 2: Sharpen (unsharp mask / Real-ESRGAN)
  → Stage 3: Color Grade (parametric curves / DPED CNN)
  → JPEG output
```

### Training the Color Grading Model

The training script uses the **DPED (DSLR Photo Enhancement Dataset)** — freely downloadable, no registration required.

**1. Download the dataset**

Go to http://people.ee.ethz.ch/~ihnatova/ and download one of the phone device archives:

| Archive | Device | Size |
|---------|--------|------|
| `iphone.tar.gz` | iPhone 3GS | ≈ 2.5 GB |
| `blackberry.tar.gz` | BlackBerry Passport | ≈ 2 GB |
| `sony.tar.gz` | Sony Xperia Z | ≈ 2.5 GB |

**2. Extract and train**

```bash
tar -xzf iphone.tar.gz   # creates dped/iphone/training_data/

python backend/training/train.py \
  --dataset /path/to/dped/iphone \
  --device iphone \
  --crop-size 100 \
  --epochs 50 \
  --batch-size 16
```

**3. Evaluate a checkpoint**

```bash
python backend/training/train.py \
  --dataset /path/to/dped/iphone \
  --device iphone \
  --eval-only \
  --checkpoint weights/colorgrade.pth
```

The trained `colorgrade.pth` file goes in `backend/weights/` and is loaded automatically when `PIPELINE_MODE=pytorch`.

## Environment Modes

- `PIPELINE_MODE=opencv` — OpenCV fallback (default, runs anywhere)
- `PIPELINE_MODE=pytorch` — Real ML models (requires GPU + model weights)

## Deployment

- **Frontend**: `cd frontend && npm run build` → deploy `dist/` to Vercel
- **Backend**: Deploy FastAPI to Modal or Render with `PIPELINE_MODE=pytorch`
