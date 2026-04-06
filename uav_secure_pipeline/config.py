from pathlib import Path

# ========= Detector =========
YOLO_MODEL_PATH = "yolov8n.pt"
DETECTION_CONF = 0.35

IMPORTANT_CLASSES = {
    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
}

ROI_EXPAND_PIXELS = 16
ROI_RATIO_MIN = 0.03
ROI_RATIO_MAX = 0.25

# ========= Video =========
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= DWT =========
WAVELET = "haar"
DWT_LEVEL = 3

# 嵌入強度設定為 15.0，以抵抗 BGR 色彩空間轉換的截斷誤差
# 現在我們同時埋藏在 H, V, D 三個子頻帶，為確保空間域波動不跨越 16 邊界，15.0 是一個安全且能 100% 抵抗轉換的數值
QIM_DELTA = 15.0

# ========= Signature =========
KEY_DIR = Path("./keys")
KEY_DIR.mkdir(parents=True, exist_ok=True)
PRIVATE_KEY_PATH = KEY_DIR / "device_private_key.pem"
PUBLIC_KEY_PATH = KEY_DIR / "device_public_key.pem"

# ========= Overlay =========
FONT_SCALE = 0.6
TEXT_THICKNESS = 2

# ========= GUI / Runtime =========
SHOW_WINDOW = True
DEFAULT_MAX_PAYLOAD_BYTES = 4096