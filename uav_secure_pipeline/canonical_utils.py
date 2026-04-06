from __future__ import annotations

import numpy as np


def canonicalize_frame_4bit(frame_bgr: np.ndarray) -> np.ndarray:
    """
    將每個像素做 floor(pixel / 16) * 16
    清除低 4 bits，只保留高 4 bits。
    """
    return ((frame_bgr.astype(np.uint16) // 16) * 16).astype(np.uint8)

def canonicalize_frame_3bit(frame_bgr: np.ndarray) -> np.ndarray:
    """
    將每個像素清除低 3 bits (`& 0xF8`)。
    等於 floor(pixel / 8) * 8。
    """
    return (frame_bgr & 0xF8).astype(np.uint8)

def robustize_frame_4bit(frame_bgr: np.ndarray) -> np.ndarray:
    """
    將像素值先除以 16 取下限，乘以 16 後再加上 8。
    這樣可以把每個像素強制定錨在 16 等分的正中間。
    用以吸收後續 DWT 浮水印過程造成的正負微小擾動，避免 `floor(p/16)` 取值跨界。
    """
    return ((frame_bgr.astype(np.uint16) // 16) * 16 + 8).astype(np.uint8)