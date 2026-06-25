import os
import urllib.request
import sys
from config import MODELS_DIR, FACE_DETECT_PROTOTXT, FACE_DETECT_MODEL, FACE_ENCODING_MODEL


FACE_DETECT_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_DETECT_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
FACE_ENCODING_MODEL_URL = "https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7"


def download_file(url, filepath, description):
    if os.path.exists(filepath):
        print(f"  [OK] {description} already exists.")
        return True
    try:
        print(f"  Downloading {description}...")
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, filepath, reporthook=_progress_hook)
        print(f"\n  [OK] {description} downloaded successfully.")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Failed to download {description}: {e}")
        return False


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded / total_size) * 100)
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {percent:.1f}% ({downloaded // 1024 // 1024}MB/{total_size // 1024 // 1024}MB)")
        sys.stdout.flush()


def download_all_models():
    print("=" * 60)
    print("  MODEL DOWNLOADER")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    results = []

    print("\n[1/3] Face Detection Prototxt...")
    results.append(download_file(FACE_DETECT_PROTOTXT_URL, FACE_DETECT_PROTOTXT, "deploy.prototxt"))

    print("\n[2/3] Face Detection Model (Caffe)...")
    results.append(download_file(FACE_DETECT_MODEL_URL, FACE_DETECT_MODEL, "res10_300x300_ssd_iter_140000.caffemodel"))

    print("\n[3/3] Face Encoding Model (OpenFace)...")
    results.append(download_file(FACE_ENCODING_MODEL_URL, FACE_ENCODING_MODEL, "openface_nn4.small2.v2.t7"))

    print("\n" + "=" * 60)
    if all(results):
        print("  ALL MODELS DOWNLOADED SUCCESSFULLY!")
    else:
        print("  SOME MODELS FAILED TO DOWNLOAD.")
        print("  Check your internet connection and try again.")
    print("=" * 60)

    return all(results)


if __name__ == "__main__":
    download_all_models()
