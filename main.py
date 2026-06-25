import cv2
import numpy as np
import time
import sys
import os
from datetime import datetime

from config import COLORS, LOGS_DIR
from face_detector import FaceDetector, FaceLandmarkDetector
from face_recognizer import FaceRecognizer, FaceDatabase
from object_detector import ObjectDetector
from webcam_manager import WebcamManager


class AIVisionSystem:
    def __init__(self):
        print("=" * 60)
        print("  AI VISION SYSTEM - Initializing...")
        print("=" * 60)

        self.webcam = WebcamManager()
        self.face_detector = FaceDetector()
        self.face_landmark_detector = FaceLandmarkDetector()
        self.face_recognizer = FaceRecognizer()
        self.object_detector = ObjectDetector()

        self.mode = "full"
        self.show_objects = True
        self.show_faces = True
        self.show_landmarks = False
        self.show_fps = True
        self.show_info = True
        self.auto_capture = False
        self.capture_cooldown = 3
        self.last_capture_time = 0
        self.detection_log = []

        self.initialized = all([
            self.face_detector.initialized,
            self.face_recognizer.initialized,
            self.object_detector.initialized,
        ])

        print("=" * 60)
        if self.initialized:
            print("  ALL MODULES LOADED SUCCESSFULLY!")
        else:
            print("  WARNING: Some modules failed to load.")
            print("  Face Detection: " + ("OK" if self.face_detector.initialized else "FAILED"))
            print("  Face Recognition: " + ("OK" if self.face_recognizer.initialized else "FAILED"))
            print("  Object Detection: " + ("OK" if self.object_detector.initialized else "FAILED"))
        print("=" * 60)

    def start(self):
        if not self.webcam.open():
            print("Cannot start without camera.")
            return
        self._print_controls()
        self._run_loop()
        self.webcam.release()
        cv2.destroyAllWindows()

    def _print_controls(self):
        print("\n" + "=" * 60)
        print("  CONTROLS (Keyboard)")
        print("=" * 60)
        print("  Q / ESC  : Quit")
        print("  S        : Save screenshot")
        print("  A        : Add face (enter name)")
        print("  R        : Remove face (enter name)")
        print("  L        : Toggle landmark detection")
        print("  O        : Toggle object detection")
        print("  F        : Toggle face detection")
        print("  I        : Toggle info panel")
        print("  T        : Toggle FPS counter")
        print("  SPACE    : Pause/Resume")
        print("  1        : Mode - Full (faces + objects)")
        print("  2        : Mode - Faces only")
        print("  3        : Mode - Objects only")
        print("=" * 60)

    def _run_loop(self):
        paused = False
        frame_count = 0

        while True:
            if not paused:
                ret, frame = self.webcam.read_frame()
                if not ret:
                    print("Failed to read frame. Retrying...")
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)
                processed, info = self._process_frame(frame)
                self._draw_ui(processed, info)

                cv2.imshow("AI Vision System", processed)
                frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                self._save_screenshot(frame if paused else processed)
            elif key == ord('a'):
                self._add_face_ui(frame)
            elif key == ord('r'):
                self._remove_face_ui()
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
                print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
            elif key == ord('o'):
                self.show_objects = not self.show_objects
                print(f"Objects: {'ON' if self.show_objects else 'OFF'}")
            elif key == ord('f'):
                self.show_faces = not self.show_faces
                print(f"Faces: {'ON' if self.show_faces else 'OFF'}")
            elif key == ord('i'):
                self.show_info = not self.show_info
            elif key == ord('t'):
                self.show_fps = not self.show_fps
            elif key == ord(' '):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key == ord('1'):
                self.mode = "full"
                print("Mode: Full")
            elif key == ord('2'):
                self.mode = "faces"
                print("Mode: Faces only")
            elif key == ord('3'):
                self.mode = "objects"
                print("Mode: Objects only")

    def _process_frame(self, frame):
        info = {
            "faces": [],
            "face_labels": [],
            "objects": [],
            "object_summary": {},
            "categories": set(),
            "fps": self.webcam.get_fps(),
        }

        if self.mode in ("full", "faces") and self.show_faces:
            faces = self.face_detector.detect(frame)
            if faces:
                encodings, labels = self.face_recognizer.recognize_frame(frame, faces)
                info["faces"] = faces
                info["face_labels"] = labels

                if self.show_landmarks:
                    landmarks = self.face_landmark_detector.detect_with_landmarks(frame)

                if self.auto_capture:
                    self._auto_capture_check(frame, faces, labels)

        if self.mode in ("full", "objects") and self.show_objects:
            detections, summary, categories = self.object_detector.detect_and_classify(frame)
            info["objects"] = detections
            info["object_summary"] = summary
            info["categories"] = categories

        return frame, info

    def _draw_ui(self, frame, info):
        if info["faces"] and self.show_faces:
            labels = info["face_labels"]
            face_labels_formatted = []
            for i, face in enumerate(info["faces"]):
                if i < len(labels):
                    lbl = labels[i]
                    if lbl != "Unknown":
                        face_labels_formatted.append({"bbox": face["bbox"], "label": lbl, "known": True})
                    else:
                        face_labels_formatted.append({"bbox": face["bbox"], "label": "Unknown", "known": False})
                else:
                    face_labels_formatted.append({"bbox": face["bbox"], "label": "Face", "known": False})

            for fl in face_labels_formatted:
                x1, y1, x2, y2 = fl["bbox"]
                color = COLORS["face_known"] if fl["known"] else COLORS["face_unknown"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = fl["label"]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["white"], 1, cv2.LINE_AA)

        if info["objects"] and self.show_objects:
            frame = self.object_detector.draw_detections(frame, info["objects"])

        if self.show_fps:
            fps_text = f"FPS: {info['fps']:.1f}"
            cv2.putText(frame, fps_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["green"], 2)

        if self.show_info:
            self._draw_info_panel(frame, info)

    def _draw_info_panel(self, frame, info):
        h, w = frame.shape[:2]
        panel_w = 280
        panel_x = w - panel_w
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 0), (w, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 25
        cv2.putText(frame, "AI VISION SYSTEM", (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["cyan"], 2)
        y += 25

        cv2.line(frame, (panel_x, y), (w, y), COLORS["cyan"], 1)
        y += 20

        cv2.putText(frame, f"Mode: {self.mode.upper()}", (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 1)
        y += 20

        n_faces = len(info["faces"])
        cv2.putText(frame, f"Faces detected: {n_faces}", (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 1)
        y += 20

        known = sum(1 for l in info["face_labels"] if l != "Unknown")
        unknown = n_faces - known
        cv2.putText(frame, f"  Known: {known}  Unknown: {unknown}", (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["green"], 1)
        y += 25

        n_objects = len(info["objects"])
        cv2.putText(frame, f"Objects detected: {n_objects}", (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 1)
        y += 20

        summary = info["object_summary"]
        for name, data in sorted(summary.items(), key=lambda x: x[1]["count"], reverse=True)[:8]:
            txt = f"  {name}: x{data['count']}"
            cv2.putText(frame, txt, (panel_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["yellow"], 1)
            y += 16

        y += 5
        cv2.line(frame, (panel_x, y), (w, y), COLORS["cyan"], 1)
        y += 15

        db = self.face_recognizer.database
        total_known = len(db.get_all_names())
        total_encodings = db.get_face_count()
        cv2.putText(frame, f"Known people: {total_known}", (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["green"], 1)
        y += 18
        cv2.putText(frame, f"Total encodings: {total_encodings}", (panel_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["green"], 1)

    def _save_screenshot(self, frame):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("captured_faces", f"screenshot_{timestamp}.jpg")
        os.makedirs("captured_faces", exist_ok=True)
        cv2.imwrite(path, frame)
        print(f"Screenshot saved: {path}")

    def _add_face_ui(self, frame):
        print("\n--- Add Face ---")
        name = input("Enter person's name: ").strip()
        if not name:
            print("Invalid name.")
            return

        faces = self.face_detector.detect(frame)
        if not faces:
            print("No face detected in current frame.")
            return

        if len(faces) > 1:
            print(f"Multiple faces detected ({len(faces)}). Using the largest one.")
            faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)

        success, msg = self.face_recognizer.add_person(name, frame, faces[0]["bbox"])
        print(msg)

    def _remove_face_ui(self):
        print("\n--- Remove Face ---")
        names = self.face_recognizer.database.get_all_names()
        if not names:
            print("No known faces in database.")
            return
        print("Known faces:", ", ".join(names))
        name = input("Enter name to remove: ").strip()
        if self.face_recognizer.database.remove_face(name):
            print(f"Removed {name} from database.")
        else:
            print(f"{name} not found.")

    def process_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Cannot read image: {image_path}")
            return None
        processed, info = self._process_frame(frame)
        return processed, info

    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps_vid, (w, h))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed, info = self._process_frame(frame)
            self._draw_ui(processed, info)
            if writer:
                writer.write(processed)
            cv2.imshow("Processing Video", processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Done. Processed {frame_count} frames.")

    def _auto_capture_check(self, frame, faces, labels):
        now = time.time()
        if now - self.last_capture_time < self.capture_cooldown:
            return
        for i, label in enumerate(labels):
            if label == "Unknown":
                self.last_capture_time = now
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                x1, y1, x2, y2 = faces[i]["bbox"]
                path = os.path.join("captured_faces", f"auto_{timestamp}.jpg")
                cv2.imwrite(path, frame[y1:y2, x1:x2])
                print(f"Auto-captured unknown face: {path}")
                break


def main():
    system = AIVisionSystem()
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--image" and len(sys.argv) > 2:
            result = system.process_image(sys.argv[2])
            if result is not None:
                processed, info = result
                cv2.imshow("Result", processed)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return
        elif arg == "--video" and len(sys.argv) > 2:
            output = sys.argv[3] if len(sys.argv) > 3 else None
            system.process_video(sys.argv[2], output)
            return
        elif arg == "--help":
            print("Usage:")
            print("  python main.py              : Live webcam mode")
            print("  python main.py --image FILE : Process image file")
            print("  python main.py --video FILE [OUTPUT]: Process video file")
            return
    system.start()


if __name__ == "__main__":
    main()
