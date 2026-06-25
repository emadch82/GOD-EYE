import os
import urllib.request
import sys
from config import MODELS_DIR


MODELS = [
    {
        "name": "YuNet Face Detector",
        "url": "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "path": os.path.join(MODELS_DIR, "yunet_2023mar.onnx"),
    },
    {
        "name": "SFace Recognition",
        "url": "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "path": os.path.join(MODELS_DIR, "sface_2021dec.onnx"),
    },
    {
        "name": "YOLOv8n Object Detection",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx",
        "path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.onnx"),
    },
]


def download_file(url, filepath, description):
    if os.path.exists(filepath):
        print(f"  [OK] {description}")
        return True
    try:
        print(f"  Downloading {description}...")
        urllib.request.urlretrieve(url, filepath, reporthook=_progress_hook)
        print(f"\n  [OK] {description} done")
        return True
    except Exception as e:
        print(f"\n  [ERROR] {description}: {e}")
        return False


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded / total_size) * 100)
        bar_len = 30
        filled = int(bar_len * percent / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {percent:.0f}%")
        sys.stdout.flush()


def download_all_models():
    print("=" * 50)
    print("  MODEL DOWNLOADER")
    print("=" * 50)
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = []
    for i, m in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] {m['name']}...")
        results.append(download_file(m["url"], m["path"], m["name"]))
    print("\n" + "=" * 50)
    if all(results):
        print("  ALL MODELS READY!")
    else:
        print("  SOME FAILED - check internet")
    print("=" * 50)
    return all(results)


if __name__ == "__main__":
    download_all_models()
