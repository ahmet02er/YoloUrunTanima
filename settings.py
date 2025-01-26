from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

IMAGE = 'Resim'
VIDEO = 'Video'
WEBCAM = 'Kamera'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]

IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'test.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'test_detected.jpg'

VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'Custom': VIDEO_DIR / 'Custom.mp4',
    'Yolo': VIDEO_DIR / 'yolo.mp4'
}


MODEL_DIR = ROOT / 'model'
DETECTION_MODEL = MODEL_DIR

DETECTION_MODEL_LIST = [
    "custom.pt",
    "yolov11n.pt"]

WEBCAM_PATH = 0
