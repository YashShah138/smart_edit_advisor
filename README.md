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

**2. Train** (run from inside `raw-enhance/`)

```bash
cd raw-enhance

# Train on iPhone pairs
python3 backend/training/train.py \
  --dataset ../dataset/iphone \
  --device iphone \
  --crop-size 100 \
  --epochs 50 \
  --batch-size 16

# Or train on BlackBerry / Sony pairs instead
python backend/training/train.py \
  --dataset ../dataset/blackberry \
  --device blackberry \
  --crop-size 100 \
  --epochs 50 \
  --batch-size 16
```

**3. Evaluate a checkpoint**

```bash
python backend/training/train.py \
  --dataset ../dataset/iphone \
  --device iphone \
  --crop-size 100 \
  --eval-only \
  --checkpoint weights/colorgrade.pth
```

The trained `colorgrade.pth` file goes in `backend/weights/` and is loaded automatically when `PIPELINE_MODE=pytorch`.

## Challenges & Fixes

### 1. Dataset access — MIT-Adobe FiveK requires an academic request form
The original plan used the MIT-Adobe FiveK dataset (5,000 RAW photos with professional retouches). It requires submitting an academic request and can take days to get approved. Switched to **DPED (DSLR Photo Enhancement Dataset)**, which is freely downloadable with no sign-up at http://people.ee.ethz.ch/~ihnatova/. DPED has ~160,000 pre-cropped 100×100 phone/DSLR image pairs, making it better suited for patch-based CNN training anyway.

### 2. Python 3.13 multiprocessing — `collate_fn` pickling error
PyTorch's `DataLoader` with `num_workers > 0` spawns child processes and pickles everything it sends to them. In Python 3.13, macOS switched from `fork` to `spawn` as the default multiprocessing start method. `spawn` requires all objects to be picklable by name — but `collate_fn` was defined as a local function *inside* `train()`, which can't be pickled. Fix: moved `collate_fn` to module level (`_collate_fn`) and set `num_workers=0` so loading runs in the main process, bypassing spawn entirely.

### 3. UNet spatial size mismatch — model output smaller than target
With 100×100 input patches, three `MaxPool2d(2)` layers halve dimensions: 100 → 50 → 25 → **12** (odd number floors). The decoder then doubles back: 12 → 24 → 48 → **96** — 4 pixels short of the 100×100 target, causing a `RuntimeError` in the L1 loss. Fix: pad the input to the next multiple of 8 (= 2³ pool stages) at the start of `forward`, process at 104×104, then crop the output back to the original 100×100. This makes the UNet resolution-agnostic.

### 4. PermissionError writing temp files
The backend tried to write temporary RAW files and session cache to the mounted workspace folder (`/mnt/editing_ai/`), which has restricted write permissions. Fix: switched all temp file paths to `tempfile.gettempdir()` (`/tmp`), which is always writable.

### 5. Frontend rollup native binary missing on Apple Silicon
`npm install` on an arm64 Mac pulled in `@rollup/rollup-linux-x64-gnu` instead of the arm64 variant, causing a `MODULE_NOT_FOUND` error at build time. Fix: `npm install @rollup/rollup-linux-arm64-gnu --save-dev`.

## Environment Modes

- `PIPELINE_MODE=opencv` — OpenCV fallback (default, runs anywhere)
- `PIPELINE_MODE=pytorch` — Real ML models (requires GPU + model weights)

## Deployment

- **Frontend**: `cd frontend && npm run build` → deploy `dist/` to Vercel
- **Backend**: Deploy FastAPI to Modal or Render with `PIPELINE_MODE=pytorch`
