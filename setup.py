import os
import sys

def setup():
    print("=" * 60)
    print("  AI VISION SYSTEM - Setup")
    print("=" * 60)

    print("\n[1] Creating directory structure...")
    dirs = [
        "models",
        "face_database",
        "captured_faces",
        "logs",
        "dataset/train",
        "dataset/test",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  Created: {d}")

    print("\n[2] Creating sample dataset structure...")
    sample_people = ["person1", "person2", "person3"]
    for person in sample_people:
        train_dir = os.path.join("dataset", "train", person)
        test_dir = os.path.join("dataset", "test", person)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        print(f"  Created directories for: {person}")

    print("\n[3] Downloading required models...")
    from model_downloader import download_all_models
    success = download_all_models()

    print("\n[4] Creating sample config file...")
    config_content = """# AI Vision System Configuration
# Edit these settings as needed

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FACE_DETECT_CONFIDENCE = 0.5
FACE_RECOGNITION_TOLERANCE = 0.5
YOLO_CONFIDENCE = 0.25
"""
    with open("config_user.py", "w") as f:
        f.write(config_content)
    print("  Created: config_user.py")

    print("\n" + "=" * 60)
    print("  SETUP COMPLETE!")
    print("=" * 60)
    print("\n  To start the system:")
    print("    python main.py")
    print("\n  To process an image:")
    print("    python main.py --image path/to/image.jpg")
    print("\n  To process a video:")
    print("    python main.py --video path/to/video.mp4")
    print("\n  To train on custom dataset:")
    print("    python trainer.py train dataset/train")
    print("\n  To run tests:")
    print("    python test_system.py")
    print("=" * 60)


if __name__ == "__main__":
    setup()
