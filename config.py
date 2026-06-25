import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FACE_DB_DIR = os.path.join(BASE_DIR, "face_database")
CAPTURED_DIR = os.path.join(BASE_DIR, "captured_faces")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

FACE_DETECT_PROTOTXT = os.path.join(MODELS_DIR, "deploy.prototxt")
FACE_DETECT_MODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
FACE_ENCODING_MODEL = os.path.join(MODELS_DIR, "openface_nn4.small2.v2.t7")

FACE_DETECT_CONFIDENCE = 0.5
FACE_RECOGNITION_TOLERANCE = 0.5
FACE_ENCODING_DIMENSION = 128

YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.25

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

KNOWN_FACES_FILE = os.path.join(FACE_DB_DIR, "known_faces.pkl")
LOG_FILE = os.path.join(LOGS_DIR, "system.log")

COLORS = {
    "face_known": (0, 255, 0),
    "face_unknown": (0, 0, 255),
    "face_detected": (255, 165, 0),
    "object": (0, 255, 255),
    "text_bg": (0, 0, 0),
    "text_fg": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "blue": (255, 0, 0),
    "cyan": (255, 255, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}
