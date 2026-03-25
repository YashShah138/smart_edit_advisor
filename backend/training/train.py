"""
Training script for the color grading CNN on the DPED dataset.

This script trains a lightweight image enhancement network that learns to map
low-quality smartphone photos to DSLR-quality images.

Dataset: DPED — DSLR Photo Enhancement Dataset
  - Freely downloadable, no sign-up required
  - URL: http://people.ee.ethz.ch/~ihnatova/
  - ~6,000 aligned pairs per device (iPhone / BlackBerry / Sony → Canon DSLR)
  - Input: smartphone JPEG photo
  - Target: Canon 70D DSLR JPEG (same scene, same moment)

Architecture: Small UNet-style network (~2M parameters)
  - Encoder: 4 conv blocks with downsampling
  - Decoder: 4 conv blocks with upsampling + skip connections
  - Output: sigmoid activation (clamped to [0, 1])

Training:
  - Loss: L1
  - Optimizer: Adam, lr=1e-4
  - Batch size: 8 (256×256 crops)
  - Epochs: 50 (converges in ~30)
  - Hardware: Google Colab T4 GPU, ~2–3 hours

Quick start:
  1. Download one of the DPED phone folders from:
       http://people.ee.ethz.ch/~ihnatova/
     e.g. download  iphone.tar.gz  (≈ 2.5 GB)
  2. Extract to a local folder so the structure is:
       dped/
       └── iphone/
           └── training_data/
               ├── iphone/        ← input JPEGs (e.g. 1.jpg, 2.jpg …)
               └── canon/         ← target JPEGs (same filenames)
  3. python3 train.py --dataset /path/to/dped/iphone --device iphone --epochs 50

Usage:
    python3 train.py --dataset /path/to/dped/iphone --device iphone --epochs 50
    python3 train.py --dataset /path/to/dped/iphone --eval-only --checkpoint weights/colorgrade.pth

Output:
    Saves model checkpoint to --output-dir (default: weights/colorgrade.pth)
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Dataset device options ────────────────────────────────────────────────────
DPED_DEVICES = ["iphone", "blackberry", "sony"]


def build_model():
    """
    Build the image enhancement UNet model.

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
        Lightweight UNet for image-to-image photo enhancement.

        Encoder: 3 → 32 → 64 → 128 → 256
        Decoder: 256 → 128 → 64 → 32 → 3
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
            self.dec3 = ConvBlock(256, 128)   # 128 up + 128 skip
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
            d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

            return self.final(d1)

    model = ColorGradeUNet()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    return model


def _find_image_files(directory: Path) -> dict:
    """Return a stem → Path mapping for all JPEG/PNG files in a directory."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    result = {}
    if not directory.exists():
        return result
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in exts:
            result[p.stem.lower()] = p
    return result


class DPEDDataset:
    """
    DPED — DSLR Photo Enhancement Dataset loader.

    Download:
        http://people.ee.ethz.ch/~ihnatova/

    Expected directory layout (after extracting one device archive):
        <dataset_dir>/
        └── training_data/
            ├── iphone/          ← or blackberry / sony
            │   ├── 1.jpg
            │   ├── 2.jpg
            │   └── ...
            └── canon/           ← DSLR targets (same filenames)
                ├── 1.jpg
                ├── 2.jpg
                └── ...

    Pass the device root as --dataset, e.g.:
        --dataset /data/dped/iphone  --device iphone

    The loader matches pairs by filename stem (e.g. "1" ↔ "1").
    """

    def __init__(self, dataset_dir: str, device: str = "iphone", crop_size: int = 256):
        self.root = Path(dataset_dir)
        self.crop_size = crop_size
        self.device = device.lower()

        if self.device not in DPED_DEVICES:
            raise ValueError(
                f"Unknown device '{device}'. Choose from: {DPED_DEVICES}\n"
                "Example: --device iphone"
            )

        train_root = self.root / "training_data"

        # Support two layouts:
        #   (A) <root>/training_data/<device>/ + <root>/training_data/canon/
        #   (B) <root>/<device>/ + <root>/canon/   (extracted flat)
        input_dir_a = train_root / self.device
        target_dir_a = train_root / "canon"
        input_dir_b = self.root / self.device
        target_dir_b = self.root / "canon"

        if input_dir_a.exists() and target_dir_a.exists():
            input_dir, target_dir = input_dir_a, target_dir_a
        elif input_dir_b.exists() and target_dir_b.exists():
            input_dir, target_dir = input_dir_b, target_dir_b
        else:
            raise FileNotFoundError(
                f"\nDPED dataset not found under: {self.root}\n"
                "\nExpected one of these layouts:\n"
                f"  {self.root}/training_data/{self.device}/  ← input phone images\n"
                f"  {self.root}/training_data/canon/          ← DSLR target images\n"
                "\nOR (flat extraction):\n"
                f"  {self.root}/{self.device}/  ← input phone images\n"
                f"  {self.root}/canon/          ← DSLR target images\n"
                "\nDownload the DPED dataset (free, no sign-up):\n"
                "  http://people.ee.ethz.ch/~ihnatova/\n"
                f"\nSelect the '{self.device}' archive (≈2–3 GB), extract it, then run:\n"
                f"  python3 train.py --dataset /path/to/dped/{self.device} "
                f"--device {self.device}"
            )

        self.input_files = _find_image_files(input_dir)
        self.target_files = _find_image_files(target_dir)

        if not self.input_files:
            raise FileNotFoundError(
                f"No images found in {input_dir}.\n"
                "Make sure the DPED archive was fully extracted."
            )
        if not self.target_files:
            raise FileNotFoundError(
                f"No DSLR target images found in {target_dir}.\n"
                "Make sure both the phone and canon folders are present."
            )

        common = sorted(set(self.input_files) & set(self.target_files))
        if not common:
            # Fallback: match by position if filenames differ
            print(
                "Warning: filename stems don't match between input and target.\n"
                "Falling back to positional matching (pairs by sort order)."
            )
            input_list = sorted(self.input_files.values())
            target_list = sorted(self.target_files.values())
            n = min(len(input_list), len(target_list))
            self.pairs = [(inp, tgt) for inp, tgt in zip(input_list[:n], target_list[:n])]
            self._positional = True
        else:
            self.pairs = [(self.input_files[s], self.target_files[s]) for s in common]
            self._positional = False

        print(
            f"DPED [{self.device}→canon]: {len(self.pairs)} paired images "
            f"(inputs: {len(self.input_files)}, targets: {len(self.target_files)})"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """Load and return a (input, target) pair as float32 [0,1] numpy arrays."""
        import cv2

        inp_path, tgt_path = self.pairs[idx]

        def load_rgb(path: Path) -> np.ndarray:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Could not read image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        input_img = load_rgb(inp_path)
        target_img = load_rgb(tgt_path)

        # Align sizes if needed (DPED patches are already aligned, but just in case)
        if input_img.shape[:2] != target_img.shape[:2]:
            target_img = cv2.resize(
                target_img,
                (input_img.shape[1], input_img.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        # Normalize to [0, 1]
        input_img /= 255.0
        target_img /= 255.0

        # Random crop
        h, w = input_img.shape[:2]
        if h >= self.crop_size and w >= self.crop_size:
            y = np.random.randint(0, h - self.crop_size + 1)
            x = np.random.randint(0, w - self.crop_size + 1)
            input_img  = input_img [y:y + self.crop_size, x:x + self.crop_size]
            target_img = target_img[y:y + self.crop_size, x:x + self.crop_size]
        else:
            # Smaller than crop size — pad with reflection
            pad_h = max(0, self.crop_size - h)
            pad_w = max(0, self.crop_size - w)
            input_img  = np.pad(input_img,  ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            target_img = np.pad(target_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        return input_img.clip(0, 1), target_img.clip(0, 1)


def train(args):
    """Main training loop."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device_torch}")

    # Build model
    model = build_model().to(device_torch)

    # Dataset
    dataset = DPEDDataset(args.dataset, device=args.device, crop_size=args.crop_size)

    # 90/10 train/val split
    n_val   = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val

    def collate_fn(batch):
        inputs  = torch.stack([torch.from_numpy(b[0]).permute(2, 0, 1) for b in batch])
        targets = torch.stack([torch.from_numpy(b[1]).permute(2, 0, 1) for b in batch])
        return inputs, targets

    train_subset = torch.utils.data.Subset(dataset, list(range(n_train)))
    val_subset   = torch.utils.data.Subset(dataset, list(range(n_train, len(dataset))))

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    # Loss and optimiser
    l1_loss   = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_torch), targets.to(device_torch)
            optimizer.zero_grad()
            loss = l1_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device_torch), targets.to(device_torch)
                val_loss += l1_loss(model(inputs), targets).item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} — "
            f"Train L1: {train_loss:.4f}, Val L1: {val_loss:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = output_dir / "colorgrade.pth"
            torch.save(model.state_dict(), str(ckpt))
            print(f"  → Saved best model (val_loss={val_loss:.4f}) → {ckpt}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'colorgrade.pth'}")


def _eval_checkpoint(args) -> None:
    """Evaluate a saved checkpoint on the validation split."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device_torch)
    state = torch.load(args.checkpoint, map_location=device_torch)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    dataset    = DPEDDataset(args.dataset, device=args.device, crop_size=args.crop_size)
    n_val      = max(1, len(dataset) // 10)
    val_indices = list(range(len(dataset) - n_val, len(dataset)))

    def collate_fn(batch):
        inputs  = torch.stack([torch.from_numpy(b[0]).permute(2, 0, 1) for b in batch])
        targets = torch.stack([torch.from_numpy(b[1]).permute(2, 0, 1) for b in batch])
        return inputs, targets

    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    l1_loss = nn.L1Loss()
    total   = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device_torch), targets.to(device_torch)
            total += l1_loss(model(inputs), targets).item()

    print(f"Validation L1 loss: {total / len(val_loader):.4f}  (n={len(val_indices)} images)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a photo enhancement CNN on the DPED dataset.\n\n"
            "Quick start:\n"
            "  1. Download the DPED dataset (free, no registration):\n"
            "       http://people.ee.ethz.ch/~ihnatova/\n"
            "     Choose one device archive, e.g. iphone.tar.gz (≈ 2.5 GB)\n\n"
            "  2. Extract it:\n"
            "       tar -xzf iphone.tar.gz  →  dped/iphone/training_data/\n\n"
            "  3. Run training:\n"
            "       python3 train.py --dataset /path/to/dped/iphone --device iphone\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help=(
            "Path to the extracted DPED device folder, e.g. /data/dped/iphone. "
            "Must contain training_data/<device>/ and training_data/canon/."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="iphone",
        choices=DPED_DEVICES,
        help="Which phone device to use as input (default: iphone)",
    )
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--crop-size",  type=int,   default=256)
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

    if args.eval_only:
        if not args.checkpoint:
            parser.error("--eval-only requires --checkpoint")
        _eval_checkpoint(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
