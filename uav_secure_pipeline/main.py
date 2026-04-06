from __future__ import annotations

import cv2
import json
import uuid
import csv
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageFont

from ceasc_lite_roi import CEASCLiteROI, draw_roi_boxes
from signature_utils import ensure_keys, sign_dict, get_device_identifier, short_id, sha256_bytes
from lsb_embed import embed_payload_lsb, check_capacity_lsb
from overlay_utils import draw_overlay
from canonical_utils import canonicalize_frame_3bit
from verify_roi_normalization import verify_identical_roi
from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    OUTPUT_DIR,
    WAVELET,
    DWT_LEVEL,
    QIM_DELTA,
    SHOW_WINDOW,
)


def roi_hash_from_frame(frame_bgr, roi_mask) -> str:
    masked = frame_bgr.copy()
    masked[roi_mask == 0] = 0
    return sha256_bytes(masked.tobytes())


def build_min_payload(frame_index: int, session_id: str, ts: str, device_id_full: str, roi_hash: str) -> bytes:
    signed_data = {
        "schema_version": 3,
        "frame_index": frame_index,
        "session_id": session_id,
        "timestamp_utc": ts,
        "device_id": device_id_full,
        "roi_hash_sha256": roi_hash,
        "canonical_rule": "floor(pixel/16)*16",
    }
    signature_b64 = sign_dict(signed_data)

    payload_dict = {
        "signed_data": signed_data,
        "signature_b64": signature_b64,
        "alg": "Ed25519",
    }
    return json.dumps(
        payload_dict,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def main():
    ensure_keys()

    detector = CEASCLiteROI()
    device_id_full = get_device_identifier()
    device_id_short = short_id(device_id_full, 12)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 20.0

    output_path = OUTPUT_DIR / f"uav_signed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
    csv_log_path = output_path.with_suffix('.csv')
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

    if not writer.isOpened():
        print("Cannot create output video")
        return

    csv_file = open(csv_log_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Frame_Index", "Timestamp", "Embed_Status", "ROI_Sync",
        "Capacity_bits", "Needed_bits", "ROI_Ratio", "PSNR_dB"
    ])

    frame_index = 0
    session_id = str(uuid.uuid4())

    print("Press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        canonical_frame = canonicalize_frame_3bit(frame)

        roi_mask, detections = detector.infer(canonical_frame)
        roi_ratio = float(roi_mask.mean())
        ts = datetime.now(timezone.utc).isoformat()

        roi_hash = roi_hash_from_frame(canonical_frame, roi_mask)

        # 只建立最小 payload
        payload_bytes = build_min_payload(
            frame_index=frame_index,
            session_id=session_id,
            ts=ts,
            device_id_full=device_id_full,
            roi_hash=roi_hash,
        )

        capacity_bits, _ = check_capacity_lsb(frame_bgr=frame, roi_mask=roi_mask)
        need_bits = (len(payload_bytes) + 4) * 8  # +4 for payload length header

        embed_status = "OK"
        vis_frame = frame

        try:
            vis_frame = embed_payload_lsb(
                frame_bgr=frame,
                roi_mask=roi_mask,
                payload=payload_bytes
            )
        except Exception as e:
            embed_status = "FAIL"
            print(f"Embed exception: {e}")
            vis_frame = frame

        # 驗證原本影像與藏完浮水印影像的 ROI 是否一致
        is_roi_consistent = verify_identical_roi(frame, vis_frame, detector)
        roi_sync_status = "OK" if is_roi_consistent else "FAIL"

        psnr_val = 0.0
        if embed_status == "OK":
            psnr_val = cv2.PSNR(frame, vis_frame)

        csv_writer.writerow([
            frame_index, ts, embed_status, roi_sync_status,
            capacity_bits, need_bits, f"{roi_ratio:.4f}", f"{psnr_val:.4f}"
        ])

        vis = draw_roi_boxes(vis_frame, detections)
        vis = draw_overlay(
            vis,
            device_id=device_id_short,
            ts=ts,
            sig_short=short_id(roi_hash, 24),
            roi_ratio=roi_ratio,
        )

        cv2.putText(
            vis,
            f"EMBED: {embed_status}  CAP(n):{capacity_bits}  NEED:{need_bits}",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if embed_status == "OK" else (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            vis,
            f"ROI SYNC: {roi_sync_status}",
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if is_roi_consistent else (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(vis_frame)
        if SHOW_WINDOW:
            cv2.imshow("UAV Secure Pipeline", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        frame_index += 1

    writer.release()
    cap.release()
    csv_file.close()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print(f"Saved to: {output_path}")
    print(f"Log saved to: {csv_log_path}")


if __name__ == "__main__":
    main()