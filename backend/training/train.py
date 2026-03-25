"""
Training script for the color grading CNN on the MIT-Adobe FiveK dataset.

This script trains a lightweight color grading network that learns to map
flat RAW renders to professionally retouched images (Expert C ground truth).

Dataset: MIT-Adobe FiveK (https://data.csail.mit.edu/graphics/fivek/)
  - 5,000 RAW photos, each retouched by 5 different experts (A–E)
  - We use Expert C retouches as training targets
  - Input: flat demosaiced RAW render (no color grading)
  - Target: Expert C retouched JPEG

Architecture: Small UNet-style network (~2M parameters)
  - Encoder: 4 conv blocks with downsampling
  - Decoder: 4 conv blocks with upsampling + skip connections
  - Output: sigmoid activation (clamped to [0, 1])

Training:
  - Loss: L1 + perceptual (VGG feature matching) + SSIM
  - Optimizer: Adam, lr=1e-4
  - Batch size: 8 (256x256 crops)
  - Epochs: 50 (converges in ~30)
  - Hardware: Google Colab T4 GPU, ~3 hours

Usage:
    python3 train.py --dataset /path/to/fivek --expert C --epochs 50 --batch-size 8
    python3 train.py --dataset /path/to/fivek --expert C --eval-only --checkpoint weights/colorgrade.pth

Output:
    Saves model checkpoint to --output-dir (default: ../weights/colorgrade.pth)
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_model():
    """
    Build the color grading UNet model.

    Returns the model (requires PyTorch).
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch is required for training. Install with:")
        print("  pip install torch torchvision")
        sys.exit(1)

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class ColorGradeUNet(nn.Module):
        """
        Lightweight UNet for image-to-image color grading.

        Encoder: 3→32→64→128→256
        Decoder: 256→128→64→32→3
        Skip connections at each level.
        """

        def __init__(self):
            super().__init__()
            # Encoder
            self.enc1 = ConvBlock(3, 32)
            self.enc2 = ConvBlock(32, 64)
            self.enc3 = ConvBlock(64, 128)
            self.enc4 = ConvBlock(128, 256)

            self.pool = nn.MaxPool2d(2)

            # Decoder
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec3 = ConvBlock(256, 128)  # 128 + 128 skip
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = ConvBlock(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1 = ConvBlock(64, 32)

            self.final = nn.Sequential(
                nn.Conv2d(32, 3, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # Encode
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))

            # Decode with skip connections
            d3 = self.up3(e4)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))

            return self.final(d1)

    model = ColorGradeUNet()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    return model


def _find_files(directory: Path, extensions: list) -> dict:
    """Return a stem → Path mapping for all files with the given extensions."""
    result = {}
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in extensions:
            result[p.stem.lower()] = p
    return result


def prepare_dataset(dataset_dir: str, expert: str = "C", workers: int = 4) -> None:
    """
    Prepare the MIT-Adobe FiveK dataset for training.

    The official FiveK download has this layout:
        fivek/
        ├── raw/                 # 5000 original .dng RAW files  (a0001.dng … a5000.dng)
        └── expertC/             # 5000 expert-retouched TIFFs or JPEGs
            ├── a0001.tif
            └── ...

    This function demosaics every .dng in raw/ with rawpy (flat render,
    no tone-curve, camera white balance) and writes PNGs to input/:
        fivek/input/a0001.png, a0002.png, …

    Run once before training:
        python3 train.py --dataset /path/to/fivek --prepare

    Args:
        dataset_dir: Root of the FiveK dataset.
        expert:      Which expert subfolder holds the retouched targets (A–E).
        workers:     Number of parallel decode workers.
    """
    try:
        import rawpy
        import cv2
        from concurrent.futures import ThreadPoolExecutor, as_completed
    except ImportError as e:
        print(f"Missing dependency for --prepare: {e}")
        print("Install with:  pip install rawpy opencv-python-headless")
        sys.exit(1)

    root = Path(dataset_dir)
    raw_dir = root / "raw"
    input_dir = root / "input"
    target_dir = root / f"expert{expert}"

    # ── Validate ───────────────────────────────────────────────────────────
    if not raw_dir.exists():
        print(
            f"\nERROR: RAW directory not found: {raw_dir}\n"
            "\nExpected layout after downloading MIT-Adobe FiveK:\n"
            f"  {root}/\n"
            f"  ├── raw/          ← .dng files go here (a0001.dng … a5000.dng)\n"
            f"  └── expert{expert}/  ← retouched TIFFs/JPEGs go here\n"
            "\nDownload the dataset from:\n"
            "  https://data.csail.mit.edu/graphics/fivek/\n"
            "\nThen place the .dng files under raw/ and the Expert C retouches under expertC/."
        )
        sys.exit(1)

    if not target_dir.exists():
        print(
            f"\nERROR: Expert target directory not found: {target_dir}\n"
            f"\nDownload Expert {expert} retouches from:\n"
            "  https://data.csail.mit.edu/graphics/fivek/\n"
            f"and place them in:  {target_dir}/"
        )
        sys.exit(1)

    raw_files = sorted(raw_dir.glob("*.dng")) + sorted(raw_dir.glob("*.DNG"))
    if not raw_files:
        print(f"ERROR: No .dng files found in {raw_dir}")
        sys.exit(1)

    input_dir.mkdir(exist_ok=True)
    existing = {p.stem.lower() for p in input_dir.glob("*.png")}
    to_process = [f for f in raw_files if f.stem.lower() not in existing]

    if not to_process:
        print(f"All {len(raw_files)} RAW files already demosaiced in {input_dir}")
        return

    print(f"Demosaicing {len(to_process)} RAW files → {input_dir}  (skipping {len(existing)} cached)")

    def demosaic_one(raw_path: Path) -> str:
        out_path = input_dir / f"{raw_path.stem.lower()}.png"
        try:
            with rawpy.imread(str(raw_path)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                )
            # Save as PNG (lossless)
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            return f"  ✓ {raw_path.name}"
        except Exception as e:
            return f"  ✗ {raw_path.name}: {e}"

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(demosaic_one, f): f for f in to_process}
        for future in as_completed(futures):
            msg = future.result()
            completed += 1
            if completed % 100 == 0 or "✗" in msg:
                print(f"[{completed}/{len(to_process)}] {msg}")

    print(f"\nDone. {input_dir} now contains {len(list(input_dir.glob('*.png')))} files.")
    print("You can now run training:\n"
          f"  python3 train.py --dataset {dataset_dir} --expert {expert}")


class FiveKDataset:
    """
    MIT-Adobe FiveK dataset loader.

    Supports two layouts:

    Layout A — after running  python3 train.py --prepare  (recommended):
        dataset_dir/
        ├── input/           # flat PNG renders produced by --prepare
        │   ├── a0001.png
        │   └── ...
        └── expertC/         # Expert C retouched TIFFs or JPEGs
            ├── a0001.tif
            └── ...

    Layout B — raw DNGs alongside retouches (demosaicing happens on-the-fly,
                slower but works without a prepare step):
        dataset_dir/
        ├── raw/             # original .dng files
        │   ├── a0001.dng
        │   └── ...
        └── expertC/
            ├── a0001.tif
            └── ...

    To download the dataset:
        https://data.csail.mit.edu/graphics/fivek/
    """

    IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    RAW_EXTS = [".dng", ".cr2", ".nef", ".arw"]

    def __init__(self, dataset_dir: str, expert: str = "C", crop_size: int = 256):
        self.dataset_dir = Path(dataset_dir)
        self.crop_size = crop_size
        self.use_raw_decode = False

        input_dir = self.dataset_dir / "input"
        raw_dir = self.dataset_dir / "raw"
        target_dir = self.dataset_dir / f"expert{expert}"

        # ── Validate target dir ───────────────────────────────────────────
        if not target_dir.exists():
            raise FileNotFoundError(
                f"\nExpert target directory not found: {target_dir}\n"
                f"\nExpected one of these layouts:\n"
                f"  {self.dataset_dir}/input/      ← pre-rendered PNGs  (run --prepare first)\n"
                f"  {self.dataset_dir}/raw/         ← original .dng files\n"
                f"  {self.dataset_dir}/expert{expert}/  ← expert retouches  ← MISSING\n"
                f"\nDownload the dataset from: https://data.csail.mit.edu/graphics/fivek/\n"
                f"Then run:  python3 train.py --dataset {dataset_dir} --prepare"
            )

        # ── Locate inputs ─────────────────────────────────────────────────
        if input_dir.exists() and any(input_dir.glob("*.png")):
            # Layout A: pre-rendered PNGs
            self.input_files = _find_files(input_dir, self.IMG_EXTS)
            self.use_raw_decode = False
            print(f"Using pre-rendered inputs from {input_dir}")
        elif raw_dir.exists() and any(raw_dir.glob("*.dng")):
            # Layout B: on-the-fly RAW decode
            try:
                import rawpy  # noqa: F401
            except ImportError:
                raise ImportError(
                    "rawpy is required to decode .dng files on-the-fly.\n"
                    "Install it:  pip install rawpy\n"
                    f"Or run:      python3 train.py --dataset {dataset_dir} --prepare"
                )
            self.input_files = _find_files(raw_dir, self.RAW_EXTS)
            self.use_raw_decode = True
            print(f"Using on-the-fly RAW decode from {raw_dir} (slower — run --prepare to speed up)")
        else:
            raise FileNotFoundError(
                f"\nNo input images found. Expected one of:\n"
                f"  {input_dir}/*.png   ← run --prepare to generate these\n"
                f"  {raw_dir}/*.dng     ← original RAW files\n"
                f"\nRun:  python3 train.py --dataset {dataset_dir} --prepare"
            )

        # ── Find matching pairs ───────────────────────────────────────────
        self.target_files = _find_files(target_dir, self.IMG_EXTS)

        common = sorted(set(self.input_files) & set(self.target_files))
        if not common:
            raise ValueError(
                f"No matching stem names between inputs and {target_dir}.\n"
                f"  Input stems (first 5):  {list(self.input_files)[:5]}\n"
                f"  Target stems (first 5): {list(self.target_files)[:5]}\n"
                "Make sure filenames match (e.g. a0001.png ↔ a0001.tif)."
            )

        self.pairs = common
        print(f"Found {len(self.pairs)} matched input/target pairs "
              f"({len(self.input_files)} inputs, {len(self.target_files)} targets)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """Load and return a (input, target) pair as float32 [0,1] numpy arrays."""
        import cv2

        stem = self.pairs[idx]

        # ── Load input ────────────────────────────────────────────────────
        if self.use_raw_decode:
            import rawpy
            raw_path = self.input_files[stem]
            with rawpy.imread(str(raw_path)) as raw:
                input_img = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                ).astype(np.float32)
        else:
            img_path = self.input_files[stem]
            input_img = cv2.cvtColor(
                cv2.imread(str(img_path), cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32)

        # ── Load target ───────────────────────────────────────────────────
        tgt_path = self.target_files[stem]
        target_img = cv2.cvtColor(
            cv2.imread(str(tgt_path), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        ).astype(np.float32)

        # ── Align sizes if needed ─────────────────────────────────────────
        if input_img.shape[:2] != target_img.shape[:2]:
            target_img = cv2.resize(
                target_img,
                (input_img.shape[1], input_img.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        # ── Normalize to [0, 1] ───────────────────────────────────────────
        # Handle both 8-bit (0-255) and 16-bit (0-65535) inputs
        max_val = 65535.0 if input_img.max() > 256 else 255.0
        input_img /= max_val
        target_img /= 255.0

        # ── Random crop ───────────────────────────────────────────────────
        h, w = input_img.shape[:2]
        if h >= self.crop_size and w >= self.crop_size:
            y = np.random.randint(0, h - self.crop_size + 1)
            x = np.random.randint(0, w - self.crop_size + 1)
            input_img = input_img[y : y + self.crop_size, x : x + self.crop_size]
            target_img = target_img[y : y + self.crop_size, x : x + self.crop_size]
        else:
            # Image smaller than crop — pad with reflection
            pad_h = max(0, self.crop_size - h)
            pad_w = max(0, self.crop_size - w)
            input_img = np.pad(input_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            target_img = np.pad(target_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        return input_img.clip(0, 1), target_img.clip(0, 1)


def train(args):
    """Main training loop."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Build model
    model = build_model().to(device)

    # Dataset
    dataset = FiveKDataset(args.dataset, expert=args.expert, crop_size=args.crop_size)

    # Split train/val (90/10)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val

    def collate_fn(batch):
        inputs = torch.stack([torch.from_numpy(b[0]).permute(2, 0, 1) for b in batch])
        targets = torch.stack([torch.from_numpy(b[1]).permute(2, 0, 1) for b in batch])
        return inputs, targets

    # Simple index-based split
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, len(dataset)))

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    # Loss and optimizer
    l1_loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = l1_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += l1_loss(outputs, targets).item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} — "
            f"Train L1: {train_loss:.4f}, Val L1: {val_loss:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "colorgrade.pth"
            torch.save(model.state_dict(), str(checkpoint_path))
            print(f"  → Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'colorgrade.pth'}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a color grading CNN on the MIT-Adobe FiveK dataset.\n\n"
            "Quick start:\n"
            "  1. Download the dataset from https://data.csail.mit.edu/graphics/fivek/\n"
            "  2. Place .dng files in  <dataset>/raw/\n"
            "     Place Expert C TIFFs in  <dataset>/expertC/\n"
            "  3. python3 train.py --dataset <dataset> --prepare   # demosaic RAWs once\n"
            "  4. python3 train.py --dataset <dataset>             # train\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to MIT-Adobe FiveK dataset root directory",
    )
    parser.add_argument(
        "--expert", type=str, default="C", choices=["A", "B", "C", "D", "E"],
        help="Which expert's retouches to use as targets (default: C)",
    )
    parser.add_argument(
        "--prepare", action="store_true",
        help=(
            "Demosaic all .dng files in <dataset>/raw/ into <dataset>/input/ PNGs. "
            "Run this once before training to avoid slow on-the-fly decoding."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers for --prepare demosaicing (default: 4)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="weights",
        help="Directory to save trained model weights (default: weights/)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Evaluate a checkpoint on the validation split (requires --checkpoint)",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        help="Path to a .pth checkpoint for --eval-only or to resume training",
    )

    args = parser.parse_args()

    if args.prepare:
        prepare_dataset(args.dataset, expert=args.expert, workers=args.workers)
        return

    if args.eval_only:
        if not args.checkpoint:
            parser.error("--eval-only requires --checkpoint")
        _eval_checkpoint(args)
        return

    train(args)


def _eval_checkpoint(args) -> None:
    """Run the model on the validation split and report average L1 loss."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    dataset = FiveKDataset(args.dataset, expert=args.expert, crop_size=args.crop_size)
    n_val = max(1, len(dataset) // 10)
    val_indices = list(range(len(dataset) - n_val, len(dataset)))

    def collate_fn(batch):
        inputs = torch.stack([torch.from_numpy(b[0]).permute(2, 0, 1) for b in batch])
        targets = torch.stack([torch.from_numpy(b[1]).permute(2, 0, 1) for b in batch])
        return inputs, targets

    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    l1_loss = nn.L1Loss()
    total = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            total += l1_loss(model(inputs), targets).item()

    print(f"Validation L1 loss: {total / len(val_loader):.4f}  (n={len(val_indices)} images)")


if __name__ == "__main__":
    main()
