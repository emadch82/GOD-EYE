import cv2
import numpy as np
import os
from config import FACE_DETECT_PROTOTXT, FACE_DETECT_MODEL, COLORS

YUNET_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sface_2021dec.onnx")


class FaceDetector:
    def __init__(self):
        self.detector = None
        self.initialized = False
        self._load()

    def _load(self):
        try:
            if os.path.exists(YUNET_MODEL):
                self.detector = cv2.FaceDetectorYN.create(
                    YUNET_MODEL, "", (320, 320), 0.6, 0.3, 5000)
                self.initialized = True
                print("[FaceDetector] YuNet loaded (99%+ accuracy)")
                return True
        except Exception as e:
            print(f"[FaceDetector] YuNet failed: {e}")

        try:
            if os.path.exists(FACE_DETECT_PROTOTXT) and os.path.exists(FACE_DETECT_MODEL):
                self.net = cv2.dnn.readNetFromCaffe(FACE_DETECT_PROTOTXT, FACE_DETECT_MODEL)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.initialized = True
                print("[FaceDetector] SSD fallback loaded")
                return True
        except Exception as e:
            print(f"[FaceDetector] SSD error: {e}")
        return False

    def detect(self, frame, confidence_threshold=0.6):
        if not self.initialized:
            return []

        if hasattr(self, 'detector') and self.detector is not None:
            return self._detect_yunet(frame, confidence_threshold)
        return self._detect_ssd(frame, confidence_threshold)

    def _detect_yunet(self, frame, threshold):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        success, faces = self.detector.detect(frame)
        if not success or faces is None:
            return []

        results = []
        for face in faces:
            x1, y1, w_f, h_f = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            conf = float(face[4])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x1 + w_f)
            y2 = min(h, y1 + h_f)
            landmarks = []
            if len(face) > 5:
                for j in range(5):
                    landmarks.append((float(face[5 + j * 2]), float(face[6 + j * 2])))
            if x2 - x1 > 30 and y2 - y1 > 30:
                results.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "landmarks": landmarks,
                })
        return results

    def _detect_ssd(self, frame, threshold):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        confidences = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 > 40 and y2 - y1 > 40:
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.4)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                results.append({
                    "bbox": (x, y, x + bw, y + bh),
                    "confidence": confidences[i],
                    "landmarks": [],
                })
        return results

    def draw_faces(self, frame, faces, labels=None):
        result = frame.copy()
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face["bbox"]
            conf = face["confidence"]
            color = COLORS["face_known"] if labels and labels[i] == "Known" else COLORS["face_detected"]
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            text = f"Face {conf:.2f}"
            cv2.putText(result, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return result


class FaceLandmarkDetector:
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.initialized = False
        self._load_cascades()

    def _load_cascades(self):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                self.eye_cascade = cv2.CascadeClassifier(eye_path)
                self.initialized = True
                return True
        except:
            pass
        return False

    def detect_with_landmarks(self, frame):
        if not self.initialized:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        results = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            results.append({
                "bbox": (int(x), int(y), int(x+w), int(y+h)),
                "eyes": [(int(ex+x), int(ey+y), int(ew), int(eh)) for (ex, ey, ew, eh) in eyes],
            })
        return results
