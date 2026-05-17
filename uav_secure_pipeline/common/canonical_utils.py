from __future__ import annotations

import numpy as np


def canonicalize_frame_3bit(frame_bgr: np.ndarray) -> np.ndarray:
    """Remove the lowest 3 bits from each channel."""
    return (frame_bgr & 0xF8).astype(np.uint8)


def canonicalize_frame_4bit(frame_bgr: np.ndarray) -> np.ndarray:
    """Remove the lowest 4 bits from each channel."""
    return (frame_bgr & 0xF0).astype(np.uint8)


def robustize_frame_4bit(frame_bgr: np.ndarray) -> np.ndarray:
    """Shift 4-bit canonicalized pixels to the center of the quantization bin."""
    base = canonicalize_frame_4bit(frame_bgr).astype(np.uint16)
    return np.clip(base + 8, 0, 255).astype(np.uint8)
