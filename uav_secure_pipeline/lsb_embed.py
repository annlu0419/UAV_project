import cv2
import numpy as np
from dwt_embed import pack_payload, unpack_payload, bytes_to_bits, bits_to_bytes

def check_capacity_lsb(frame_bgr: np.ndarray, roi_mask: np.ndarray) -> tuple:
    """
    估算 LSB 藏入的可用容量與所需容量。
    回傳 (capacity_bits, need_bits)
    """
    non_roi_pixels = np.count_nonzero(roi_mask == 0)
    # B, G, R 共 3 個通道，每個通道利用最低 3 bits -> 每顆像素 9 bits
    capacity_bits = non_roi_pixels * 3 * 3
    # 預估 payload roughly takes JSON string + 256 sig = ~500*8 = 4000 bits
    need_bits = 4500 
    return capacity_bits, need_bits

def embed_payload_lsb(frame_bgr: np.ndarray, roi_mask: np.ndarray, payload: bytes) -> np.ndarray:
    """
    將 payload 位元藏入非 ROI 區的 B,G,R 最低 3 bits。
    """
    out = frame_bgr.copy()
    packed = pack_payload(payload)
    bits = bytes_to_bits(packed)
    
    capacity_bits, _ = check_capacity_lsb(frame_bgr, roi_mask)
    if len(bits) > capacity_bits:
        print(f"[Warning] Payload size ({len(bits)} bits) exceeds capacity ({capacity_bits} bits). Truncating.")
        bits = bits[:capacity_bits]
        
    valid_pixels = out[roi_mask == 0]
    if len(valid_pixels) == 0 or len(bits) == 0:
        return out
        
    flat_channels = valid_pixels.reshape(-1)
    
    # 補齊 bit 長度為 3 的倍數，以利組合
    bits_pad = bits.copy()
    if len(bits) % 3 != 0:
        bits_pad = np.concatenate([bits, np.zeros(3 - (len(bits) % 3), dtype=np.uint8)])
        
    # 每 3 個 bit 壓成一個數字 (0~7)
    bits_reshaped = bits_pad.reshape(-1, 3)
    vals = (bits_reshaped[:, 0] << 2) | (bits_reshaped[:, 1] << 1) | bits_reshaped[:, 2]
    
    num_vals = len(vals)
    # 將原像素的顏色清掉最低 3 bits (& 0xF8)，並把我們製作的 vals 放進去
    flat_channels[:num_vals] = (flat_channels[:num_vals] & 0xF8) | vals
    
    # 放回原影像
    out[roi_mask == 0] = flat_channels.reshape(-1, 3)
    return out

def extract_payload_lsb(frame_bgr: np.ndarray, roi_mask: np.ndarray) -> bytes:
    """
    從非 ROI 區的最末 3 bits 提取 payload。
    """
    valid_pixels = frame_bgr[roi_mask == 0]
    if len(valid_pixels) == 0:
        raise ValueError("capacity is 0")
        
    flat_channels = valid_pixels.reshape(-1)
    
    # 取出所有 valid_pixels 的最低 3 bits
    vals = flat_channels & 0x07
    
    # 將每個 0~7 的數字拆解回 3 個獨立的 bits
    b0 = (vals >> 2) & 1
    b1 = (vals >> 1) & 1
    b2 = vals & 1
    bits = np.column_stack((b0, b1, b2)).reshape(-1).astype(np.uint8)
    
    # 轉換成 bytes
    raw = bits_to_bytes(bits)
    # 解析並解包 Payload (含有 payload 尺寸欄位防呆，會自動取需要的長度)
    return unpack_payload(raw)
