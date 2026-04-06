from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    DETECTION_CONF,
    IMPORTANT_CLASSES,
    ROI_EXPAND_PIXELS,
    ROI_RATIO_MIN,
    ROI_RATIO_MAX,
)


@dataclass
class DetectionBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls_id: int
    cls_name: str


class CEASCLiteROI:
    def __init__(self, model_path: str = YOLO_MODEL_PATH, conf: float = DETECTION_CONF):
        self.model = YOLO(model_path)
        self.conf = conf

    @staticmethod
    def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        return x1, y1, x2, y2

    def _expand_box(self, box: DetectionBox, w: int, h: int) -> Tuple[int, int, int, int]:
        d = ROI_EXPAND_PIXELS
        return self._clip_box(box.x1 - d, box.y1 - d, box.x2 + d, box.y2 + d, w, h)

    def _adaptive_adjust(self, mask: np.ndarray) -> np.ndarray:
        ratio = float(mask.mean())
        k = np.ones((3, 3), np.uint8)

        adjusted = mask.copy()
        if ratio > ROI_RATIO_MAX:
            adjusted = cv2.erode(adjusted, k, iterations=1)
        elif ratio < ROI_RATIO_MIN:
            adjusted = cv2.dilate(adjusted, k, iterations=1)

        return adjusted

    def infer(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[DetectionBox]]:
        h, w = frame_bgr.shape[:2]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        detections: List[DetectionBox] = []

        # 將影像正規化 (除以 8 取下限再乘以 8)，即像素值忽略最低 3 bits
        # 這樣可確保藏 LSB 浮水印前後丟給 YOLO 的影像完全一致，進而取得相同的 ROI
        norm_frame = (frame_bgr // 8) * 8
        
        results = self.model.predict(source=norm_frame, conf=self.conf, verbose=False)

        if not results:
            return roi_mask, detections

        r = results[0]
        names = r.names

        if r.boxes is None or len(r.boxes) == 0:
            return roi_mask, detections

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for xyxy, conf, cls_id in zip(boxes_xyxy, confs, clss):
            cls_name = str(names[int(cls_id)])
            if cls_name not in IMPORTANT_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, xyxy.tolist())
            x1, y1, x2, y2 = self._clip_box(x1, y1, x2, y2, w, h)

            det = DetectionBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                conf=float(conf), cls_id=int(cls_id), cls_name=cls_name
            )
            detections.append(det)

            ex1, ey1, ex2, ey2 = self._expand_box(det, w, h)
            roi_mask[ey1:ey2 + 1, ex1:ex2 + 1] = 1

        roi_mask = self._adaptive_adjust(roi_mask)
        return roi_mask, detections


def draw_roi_boxes(frame_bgr: np.ndarray, detections: List[DetectionBox]) -> np.ndarray:
    out = frame_bgr.copy()
    for d in detections:
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 255), 2)
        cv2.putText(
            out,
            f"{d.cls_name} {d.conf:.2f}",
            (d.x1, max(20, d.y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out