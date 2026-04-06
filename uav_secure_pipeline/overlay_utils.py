from __future__ import annotations

import cv2
import numpy as np
from config import FONT_SCALE, TEXT_THICKNESS


def draw_overlay(frame: np.ndarray, device_id: str, ts: str, sig_short: str, roi_ratio: float) -> np.ndarray:
    out = frame.copy()

    lines = [
        f"ID: {device_id}",
        f"TS: {ts}",
        f"SIG: {sig_short}",
        f"ROI ratio: {roi_ratio:.3f}",
    ]

    y = 28
    for line in lines:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (0, 255, 255),
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        y += 28

    return out