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
        self._multi_scale = False
        self._flip_augment = False
        self._contrast_enhance = False
        self._frame_count = 0
        self._cached_faces = []
        self._cache_interval = 4
        self._load()

    def _load(self):
        try:
            if os.path.exists(YUNET_MODEL):
                self.detector = cv2.FaceDetectorYN.create(
                    YUNET_MODEL, "", (320, 320), 0.4, 0.3, 5000)
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

    def detect(self, frame, confidence_threshold=0.4):
        if not self.initialized:
            return []

        self._frame_count += 1

        if self._frame_count % self._cache_interval != 0:
            return self._cached_faces

        if hasattr(self, 'detector') and self.detector is not None:
            self._cached_faces = self._detect_enhanced(frame, confidence_threshold)
            return self._cached_faces
        return self._detect_ssd(frame, confidence_threshold)

    def _enhance_for_detection(self, frame):
        enhanced = frame.copy()
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def _detect_yunet_single(self, frame, threshold):
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
            if x2 - x1 > 20 and y2 - y1 > 20:
                results.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "landmarks": landmarks,
                })
        return results

    def _nms_faces(self, faces, iou_threshold=0.3):
        if not faces:
            return []
        boxes = np.array([[f["bbox"][0], f["bbox"][1],
                           f["bbox"][2] - f["bbox"][0],
                           f["bbox"][3] - f["bbox"][1]] for f in faces])
        scores = np.array([f["confidence"] for f in faces])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [faces[i] for i in keep]

    def _detect_enhanced(self, frame, threshold):
        all_faces = []

        all_faces.extend(self._detect_yunet_single(frame, threshold))

        if self._contrast_enhance:
            enhanced = self._enhance_for_detection(frame)
            all_faces.extend(self._detect_yunet_single(enhanced, threshold))

        if self._flip_augment:
            flipped = cv2.flip(frame, 1)
            flip_faces = self._detect_yunet_single(flipped, threshold)
            fw = frame.shape[1]
            for f in flip_faces:
                x1, y1, x2, y2 = f["bbox"]
                f["bbox"] = (fw - x2, y1, fw - x1, y2)
                if f["landmarks"]:
                    f["landmarks"] = [(fw - lx, ly) for lx, ly in f["landmarks"]]
            all_faces.extend(flip_faces)

        if self._multi_scale:
            for scale in [0.75]:
                sh = int(frame.shape[0] * scale)
                sw = int(frame.shape[1] * scale)
                if sw < 100 or sh < 100:
                    continue
                scaled = cv2.resize(frame, (sw, sh))
                scale_faces = self._detect_yunet_single(scaled, threshold)
                for f in scale_faces:
                    x1, y1, x2, y2 = f["bbox"]
                    f["bbox"] = (
                        int(x1 / scale),
                        int(y1 / scale),
                        int(x2 / scale),
                        int(y2 / scale)
                    )
                    if f["landmarks"]:
                        f["landmarks"] = [(lx / scale, ly / scale) for lx, ly in f["landmarks"]]
                all_faces.extend(scale_faces)

        merged = self._nms_faces(all_faces, iou_threshold=0.3)
        return merged

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
                if x2 - x1 > 20 and y2 - y1 > 20:
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
