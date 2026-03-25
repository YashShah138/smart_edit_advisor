"""
Enhancement profile definitions.

Each profile defines parametric color curve adjustments:
- Shadow lift/crush
- Highlight roll-off
- Midtone contrast
- Saturation adjustment
- Color temperature shift
- Split-toning (shadow/highlight tint)
- Tone curve points
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from backend.api.schemas import ProfileInfo


@dataclass
class ToneCurve:
    """Bezier-style tone curve defined by control points."""
    points: List[Tuple[float, float]] = field(default_factory=lambda: [(0, 0), (1, 1)])


@dataclass
class ColorTint:
    """RGB color tint with intensity."""
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    intensity: float = 0.0


@dataclass
class ProfileParams:
    """Complete set of color grading parameters for a profile."""

    # Tone curve (input → output mapping for luminance)
    tone_curve: ToneCurve = field(default_factory=ToneCurve)

    # Per-channel curves (optional, for cross-processing effects)
    red_curve: ToneCurve = field(default_factory=ToneCurve)
    green_curve: ToneCurve = field(default_factory=ToneCurve)
    blue_curve: ToneCurve = field(default_factory=ToneCurve)

    # Basic adjustments
    exposure: float = 0.0        # EV stops (-2.0 to +2.0)
    contrast: float = 0.0        # (-1.0 to +1.0)
    saturation: float = 0.0      # (-1.0 to +1.0)
    vibrance: float = 0.0        # (-1.0 to +1.0)
    temperature: float = 0.0     # Kelvin shift (-50 to +50, warm positive)
    tint: float = 0.0            # Green-magenta (-50 to +50)

    # Shadow / highlight recovery
    shadow_lift: float = 0.0     # (0.0 to 0.3)
    highlight_rolloff: float = 0.0  # (0.0 to 0.3)

    # Split toning
    shadow_tint: ColorTint = field(default_factory=ColorTint)
    highlight_tint: ColorTint = field(default_factory=ColorTint)

    # Clarity (local contrast)
    clarity: float = 0.0         # (-1.0 to +1.0)

    # Black & white conversion
    is_bw: bool = False
    bw_red_weight: float = 0.3
    bw_green_weight: float = 0.59
    bw_blue_weight: float = 0.11

    # Fade (lifted blacks)
    fade: float = 0.0            # (0.0 to 0.2)

    # Grain
    grain: float = 0.0           # (0.0 to 1.0)


# ── Profile Definitions ──────────────────────────────────────────────────────

EXPERT_NATURAL = ProfileParams(
    tone_curve=ToneCurve(points=[(0, 0), (0.25, 0.22), (0.5, 0.5), (0.75, 0.78), (1, 1)]),
    exposure=0.1,
    contrast=0.1,
    saturation=0.05,
    vibrance=0.15,
    temperature=2.0,
    shadow_lift=0.02,
    highlight_rolloff=0.02,
    clarity=0.15,
)

WARM_FILM = ProfileParams(
    tone_curve=ToneCurve(points=[(0, 0.05), (0.25, 0.22), (0.5, 0.52), (0.75, 0.82), (1, 0.95)]),
    red_curve=ToneCurve(points=[(0, 0.02), (0.5, 0.53), (1, 1)]),
    green_curve=ToneCurve(points=[(0, 0), (0.5, 0.48), (1, 0.97)]),
    blue_curve=ToneCurve(points=[(0, 0.03), (0.5, 0.45), (1, 0.90)]),
    exposure=0.15,
    contrast=0.05,
    saturation=-0.1,
    vibrance=0.1,
    temperature=15.0,
    shadow_lift=0.08,
    highlight_rolloff=0.05,
    shadow_tint=ColorTint(r=0.15, g=0.08, b=0.0, intensity=0.25),
    highlight_tint=ColorTint(r=0.1, g=0.08, b=0.0, intensity=0.15),
    fade=0.06,
    grain=0.15,
    clarity=0.05,
)

MOODY_CONTRAST = ProfileParams(
    tone_curve=ToneCurve(points=[(0, 0), (0.15, 0.05), (0.4, 0.35), (0.6, 0.7), (0.85, 0.9), (1, 1)]),
    blue_curve=ToneCurve(points=[(0, 0.05), (0.5, 0.47), (1, 0.92)]),
    exposure=-0.1,
    contrast=0.35,
    saturation=-0.2,
    vibrance=-0.05,
    temperature=-5.0,
    shadow_lift=0.0,
    highlight_rolloff=0.08,
    shadow_tint=ColorTint(r=0.0, g=0.02, b=0.08, intensity=0.2),
    clarity=0.3,
)

BW_FINE_ART = ProfileParams(
    is_bw=True,
    bw_red_weight=0.35,
    bw_green_weight=0.50,
    bw_blue_weight=0.15,
    tone_curve=ToneCurve(points=[(0, 0.02), (0.2, 0.12), (0.5, 0.52), (0.8, 0.9), (1, 0.98)]),
    contrast=0.25,
    clarity=0.4,
    shadow_lift=0.03,
    highlight_rolloff=0.03,
    grain=0.1,
)

GOLDEN_HOUR = ProfileParams(
    tone_curve=ToneCurve(points=[(0, 0.02), (0.25, 0.20), (0.5, 0.55), (0.75, 0.82), (1, 0.97)]),
    red_curve=ToneCurve(points=[(0, 0.03), (0.5, 0.55), (1, 1)]),
    green_curve=ToneCurve(points=[(0, 0.01), (0.5, 0.50), (1, 0.95)]),
    blue_curve=ToneCurve(points=[(0, 0), (0.5, 0.40), (1, 0.85)]),
    exposure=0.2,
    contrast=0.15,
    saturation=0.15,
    vibrance=0.2,
    temperature=25.0,
    shadow_lift=0.06,
    highlight_rolloff=0.04,
    shadow_tint=ColorTint(r=0.12, g=0.05, b=0.0, intensity=0.3),
    highlight_tint=ColorTint(r=0.2, g=0.12, b=0.0, intensity=0.25),
    clarity=0.1,
)

CLEAN_COMMERCIAL = ProfileParams(
    tone_curve=ToneCurve(points=[(0, 0), (0.25, 0.27), (0.5, 0.52), (0.75, 0.77), (1, 1)]),
    exposure=0.25,
    contrast=0.15,
    saturation=0.05,
    vibrance=0.2,
    temperature=0.0,
    shadow_lift=0.0,
    highlight_rolloff=0.0,
    clarity=0.35,
)

# ── Profile Registry ─────────────────────────────────────────────────────────

PROFILES: Dict[str, ProfileParams] = {
    "expert_natural": EXPERT_NATURAL,
    "warm_film": WARM_FILM,
    "moody_contrast": MOODY_CONTRAST,
    "bw_fine_art": BW_FINE_ART,
    "golden_hour": GOLDEN_HOUR,
    "clean_commercial": CLEAN_COMMERCIAL,
}

PROFILE_INFO: List[ProfileInfo] = [
    ProfileInfo(
        id="expert_natural",
        name="Expert Natural",
        description="Clean, neutral edit closest to a professional hand-retouched look",
        aesthetic="Balanced tones, subtle warmth, natural skin tones",
    ),
    ProfileInfo(
        id="warm_film",
        name="Warm Film",
        description="Lifted shadows, warm midtones, slight highlight fade",
        aesthetic="Nostalgic film stock, golden warmth, soft grain",
    ),
    ProfileInfo(
        id="moody_contrast",
        name="Moody Contrast",
        description="Deep blacks, punchy midtones, desaturated highlights",
        aesthetic="Cinematic drama, cool shadows, high contrast",
    ),
    ProfileInfo(
        id="bw_fine_art",
        name="B&W Fine Art",
        description="Luminosity-based black and white with high local contrast",
        aesthetic="Gallery-grade monochrome, rich tonal range",
    ),
    ProfileInfo(
        id="golden_hour",
        name="Golden Hour",
        description="Warm orange grade, glowing highlights, rich shadows",
        aesthetic="Sunset warmth, golden glow, romantic mood",
    ),
    ProfileInfo(
        id="clean_commercial",
        name="Clean Commercial",
        description="Bright, neutral, high clarity for product and portrait photography",
        aesthetic="Studio-clean, crisp detail, vibrant but controlled",
    ),
]


def get_profile(profile_id: str) -> ProfileParams:
    """Get profile parameters by ID."""
    if profile_id not in PROFILES:
        raise ValueError(f"Unknown profile: {profile_id}. Available: {list(PROFILES.keys())}")
    return PROFILES[profile_id]
