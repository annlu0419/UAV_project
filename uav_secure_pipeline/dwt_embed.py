from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pywt
from typing import Tuple

DEBUG_JSON_PATH = Path("output/dwt_debug.json")


def bytes_to_bits(data: bytes) -> np.ndarray:
    bits = []
    for b in data:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
    return np.array(bits, dtype=np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.astype(np.uint8).flatten()
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    out = bytearray()
    for i in range(0, len(bits), 8):
        value = 0
        for j in range(8):
            value = (value << 1) | int(bits[i + j])
        out.append(value)
    return bytes(out)


def pack_payload(payload: bytes) -> bytes:
    return len(payload).to_bytes(4, byteorder="big") + payload


def unpack_payload(data: bytes) -> bytes:
    if len(data) < 4:
        raise ValueError("payload too short")
    n = int.from_bytes(data[:4], byteorder="big")
    body = data[4:4 + n]
    if len(body) != n:
        raise ValueError("payload length mismatch")
    return body


def project_mask_to_hh3(roi_mask: np.ndarray, hh3_shape: Tuple[int, int]) -> np.ndarray:
    h3, w3 = hh3_shape
    resized = cv2.resize(roi_mask.astype(np.uint8), (w3, h3), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def estimate_nonroi_hh3_capacity_bits(frame_bgr: np.ndarray, roi_mask: np.ndarray,
                                      wavelet: str = "haar", level: int = 3) -> int:
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)

    coeffs = pywt.wavedec2(y, wavelet=wavelet, level=level)
    cH, cV, cD = coeffs[1]

    hh3_roi_mask = project_mask_to_hh3(roi_mask, cD.shape)
    allow_mask = (hh3_roi_mask == 0).astype(np.uint8)
    return int(np.count_nonzero(allow_mask)) * 3


def embed_bits_in_coeffs(coeff: np.ndarray, allow_mask: np.ndarray, bits: np.ndarray, delta: float, debug: bool = False) -> np.ndarray:
    out = coeff.copy().astype(np.float32)
    coords = np.argwhere(allow_mask > 0)

    for i, bit in enumerate(bits):
        y, x = coords[i]
        c = out[y, x]
        q = round(c / delta)

        if (q % 2) != int(bit):
            q = q + 1 if c >= 0 else q - 1

        out[y, x] = q * delta

    if debug:
        DEBUG_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_bits": len(bits),
            "coords": [[int(c[0]), int(c[1])] for c in coords[:len(bits)]],
            "bits": [int(b) for b in bits]
        }
        DEBUG_JSON_PATH.write_text(json.dumps(data))
        print(f"[DEBUG EMBED] Saved {len(bits)} bits and coords to {DEBUG_JSON_PATH}")

    return out


def extract_bits_from_coeffs(coeff: np.ndarray, allow_mask: np.ndarray, n_bits: int, delta: float, debug: bool = False) -> np.ndarray:
    coords = np.argwhere(allow_mask > 0)
    if len(coords) < n_bits:
        raise ValueError("not enough coefficients for extraction")

    bits = []
    for i in range(n_bits):
        y, x = coords[i]
        c = coeff[y, x]
        q = round(c / delta)
        bits.append(q % 2)

    extracted_bits = np.array(bits, dtype=np.uint8)

    if debug:
        if DEBUG_JSON_PATH.exists():
            data = json.loads(DEBUG_JSON_PATH.read_text())
            exp_num = data["num_bits"]
            exp_coords = data["coords"]
            exp_bits = data["bits"]
            
            act_coords = [[int(c[0]), int(c[1])] for c in coords[:n_bits]]
            act_bits = [int(b) for b in extracted_bits]
            
            print(f"\n[DEBUG VERIFY]")
            print(f"  - Coordinates Match: {act_coords == exp_coords} ({len(act_coords)} vs {len(exp_coords)})")
            
            match_count = sum(1 for a, b in zip(exp_bits, act_bits) if a == b)
            print(f"  - Bits Match: {match_count == exp_num} ({match_count}/{exp_num})")
            
            if act_bits != exp_bits:
                print(f"  - WARN: Bit mismatches exist!")
            print("")
        else:
            print("\n[DEBUG VERIFY] No debug JSON found to compare.\n")

    return extracted_bits


def embed_payload_nonroi_hh3(frame_bgr: np.ndarray, roi_mask: np.ndarray, payload: bytes,
                             wavelet: str = "haar", level: int = 3, delta: float = 2.0, debug_first_frame: bool = False) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)

    packed = pack_payload(payload)
    bits = bytes_to_bits(packed)

    coeffs = pywt.wavedec2(y, wavelet=wavelet, level=level)
    cH, cV, cD = coeffs[1]

    hh3_roi_mask = project_mask_to_hh3(roi_mask, cD.shape)
    allow_mask = (hh3_roi_mask == 0).astype(np.uint8)

    cap_per_sub = int(np.count_nonzero(allow_mask))
    n_capacity = cap_per_sub * 3

    if len(bits) >= n_capacity:
        embed_bits = bits[:n_capacity]
    else:
        embed_bits = np.concatenate([bits, np.zeros(n_capacity - len(bits), dtype=np.uint8)])

    bits_H = embed_bits[0 : cap_per_sub]
    bits_V = embed_bits[cap_per_sub : cap_per_sub * 2]
    bits_D = embed_bits[cap_per_sub * 2 : cap_per_sub * 3]

    cH_new = embed_bits_in_coeffs(cH, allow_mask, bits_H, delta, debug=False)
    cV_new = embed_bits_in_coeffs(cV, allow_mask, bits_V, delta, debug=False)
    cD_new = embed_bits_in_coeffs(cD, allow_mask, bits_D, delta, debug=debug_first_frame)

    coeffs[1] = (cH_new, cV_new, cD_new)

    y_new = pywt.waverec2(coeffs, wavelet=wavelet)
    y_new = np.clip(y_new, 0, 255).astype(np.uint8)

    out = ycrcb.copy()
    out[:, :, 0] = y_new[:out.shape[0], :out.shape[1]]
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def extract_payload_nonroi_hh3(frame_bgr: np.ndarray, roi_mask: np.ndarray,
                               wavelet: str = "haar", level: int = 3, delta: float = 2.0, debug_first_frame: bool = False) -> bytes:
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)

    coeffs = pywt.wavedec2(y, wavelet=wavelet, level=level)
    cH, cV, cD = coeffs[1]

    hh3_roi_mask = project_mask_to_hh3(roi_mask, cD.shape)
    allow_mask = (hh3_roi_mask == 0).astype(np.uint8)

    cap_per_sub = int(np.count_nonzero(allow_mask))
    if cap_per_sub == 0:
        raise ValueError("capacity is 0")

    bits_H = extract_bits_from_coeffs(cH, allow_mask, cap_per_sub, delta, debug=False)
    bits_V = extract_bits_from_coeffs(cV, allow_mask, cap_per_sub, delta, debug=False)
    bits_D = extract_bits_from_coeffs(cD, allow_mask, cap_per_sub, delta, debug=debug_first_frame)

    bits = np.concatenate([bits_H, bits_V, bits_D])
    raw = bits_to_bytes(bits)
    return unpack_payload(raw)