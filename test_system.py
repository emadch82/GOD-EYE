import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from face_detector import FaceDetector, FaceLandmarkDetector
from face_recognizer import FaceRecognizer, FaceDatabase, FaceEncoder
from object_detector import ObjectDetector
from webcam_manager import WebcamManager
import cv2


def test_all():
    print("=" * 60)
    print("  AI VISION SYSTEM - Component Test")
    print("=" * 60)

    results = {}

    print("\n[1] Testing Face Detector...")
    fd = FaceDetector()
    results["Face Detector"] = fd.initialized

    print("\n[2] Testing Face Encoder...")
    fe = FaceEncoder()
    results["Face Encoder"] = fe.initialized

    print("\n[3] Testing Face Recognizer...")
    fr = FaceRecognizer()
    results["Face Recognizer"] = fr.initialized

    print("\n[4] Testing Face Database...")
    db = FaceDatabase()
    results["Face Database"] = True
    print(f"  Known faces: {len(db.get_all_names())}")

    print("\n[5] Testing Object Detector...")
    od = ObjectDetector()
    results["Object Detector"] = od.initialized

    print("\n[6] Testing Webcam...")
    wm = WebcamManager()
    cam_ok = wm.open()
    results["Webcam"] = cam_ok
    if cam_ok:
        ret, frame = wm.read_frame()
        results["Frame Capture"] = ret
        wm.release()

    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        symbol = "[OK]" if ok else "[!!]"
        print(f"  {symbol} {name}: {status}")
        if not ok:
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("  ALL TESTS PASSED!")
    else:
        print("  SOME TESTS FAILED.")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    test_all()
