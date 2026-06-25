import cv2
import numpy as np
import os
import pickle
from config import (
    FACE_ENCODING_MODEL, FACE_DETECT_PROTOTXT, FACE_DETECT_MODEL,
    FACE_RECOGNITION_TOLERANCE, FACE_ENCODING_DIMENSION,
    KNOWN_FACES_FILE, FACE_DB_DIR, CAPTURED_DIR, COLORS
)

TRAINED_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models", "face_recognition_model.pkl")


class MLFaceRecognizer:
    def __init__(self):
        self.model_data = None
        self.classifier = None
        self.pca = None
        self.scaler = None
        self.label_encoder = None
        self.target_names = None
        self.initialized = False
        self.detect_net = None
        self._load_trained_model()

    def _load_trained_model(self):
        if not os.path.exists(TRAINED_MODEL_PATH):
            print("[MLFaceRecognizer] Trained model not found. Run ml_trainer.py first.")
            return False
        try:
            with open(TRAINED_MODEL_PATH, "rb") as f:
                self.model_data = pickle.load(f)
            self.classifier = self.model_data["model"]
            self.pca = self.model_data["pca"]
            self.scaler = self.model_data["scaler"]
            self.label_encoder = self.model_data["label_encoder"]
            self.target_names = self.model_data["target_names"]

            if os.path.exists(FACE_DETECT_PROTOTXT) and os.path.exists(FACE_DETECT_MODEL):
                self.detect_net = cv2.dnn.readNetFromCaffe(FACE_DETECT_PROTOTXT, FACE_DETECT_MODEL)

            self.initialized = True
            print(f"[MLFaceRecognizer] Trained model loaded: {self.model_data['best_model_name']}")
            print(f"[MLFaceRecognizer] Accuracy: {self.model_data['results'][self.model_data['best_model_name']]['accuracy']:.4f}")
            print(f"[MLFaceRecognizer] People: {list(self.target_names)}")
            return True
        except Exception as e:
            print(f"[MLFaceRecognizer] Error loading model: {e}")
            return False

    def _preprocess_face(self, frame, face_bbox):
        x1, y1, x2, y2 = face_bbox
        face_region = frame[y1:y2, x1:x2]
        if face_region.size == 0:
            return None

        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (47, 62))
        face_flat = face_resized.flatten().astype(np.float32)
        return face_flat

    def predict(self, frame, face_bbox):
        if not self.initialized:
            return "Unknown", 0.0, []

        face_vector = self._preprocess_face(frame, face_bbox)
        if face_vector is None:
            return "Unknown", 0.0, []

        face_scaled = self.scaler.transform(face_vector.reshape(1, -1))
        face_pca = self.pca.transform(face_scaled)
        prediction = self.classifier.predict(face_pca)[0]

        name = self.label_encoder.inverse_transform([prediction])[0]

        probabilities = []
        if hasattr(self.classifier, 'predict_proba'):
            proba = self.classifier.predict_proba(face_pca)[0]
            for i, prob in enumerate(proba):
                label = self.label_encoder.inverse_transform([i])[0]
                probabilities.append({"name": label, "confidence": float(prob)})
            probabilities.sort(key=lambda x: x["confidence"], reverse=True)

        confidence = 0.0
        if probabilities:
            confidence = probabilities[0]["confidence"]

        return name, confidence, probabilities

    def recognize_faces(self, frame, faces):
        results = []
        for face in faces:
            name, confidence, probs = self.predict(frame, face["bbox"])
            results.append({
                "bbox": face["bbox"],
                "name": name,
                "confidence": confidence,
                "probabilities": probs[:5],
                "detection_confidence": face.get("confidence", 0.0)
            })
        return results

    def get_lfw_names(self):
        if self.target_names is not None:
            return list(self.target_names)
        return []


class FaceEncoder:
    def __init__(self):
        self.encoding_net = None
        self.detect_net = None
        self.initialized = False
        self._load_models()

    def _load_models(self):
        try:
            if os.path.exists(FACE_ENCODING_MODEL):
                self.encoding_net = cv2.dnn.readNetFromTorch(FACE_ENCODING_MODEL)
                print("[FaceEncoder] Encoding model loaded.")

            if os.path.exists(FACE_DETECT_PROTOTXT) and os.path.exists(FACE_DETECT_MODEL):
                self.detect_net = cv2.dnn.readNetFromCaffe(FACE_DETECT_PROTOTXT, FACE_DETECT_MODEL)
                print("[FaceEncoder] Detection model loaded.")

            if self.encoding_net and self.detect_net:
                self.initialized = True
                print("[FaceEncoder] All models loaded successfully.")
                return True
        except Exception as e:
            print(f"[FaceEncoder] Error loading models: {e}")
        return False

    def _align_face(self, frame, face_bbox):
        x1, y1, x2, y2 = face_bbox
        face_region = frame[y1:y2, x1:x2]
        if face_region.size == 0:
            return None
        face_resized = cv2.resize(face_region, (96, 96))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        return face_rgb

    def encode_face(self, frame, face_bbox):
        if not self.initialized:
            return None
        aligned = self._align_face(frame, face_bbox)
        if aligned is None:
            return None
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.encoding_net.setInput(blob)
        encoding = self.encoding_net.forward()
        encoding = encoding.flatten()
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding = encoding / norm
        return encoding

    def encode_faces_in_frame(self, frame, faces):
        encodings = []
        for face in faces:
            encoding = self.encode_face(frame, face["bbox"])
            encodings.append(encoding)
        return encodings


class FaceDatabase:
    def __init__(self):
        self.known_names = []
        self.known_encodings = []
        self.known_metadata = []
        self._load_database()

    def _load_database(self):
        if os.path.exists(KNOWN_FACES_FILE):
            try:
                with open(KNOWN_FACES_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.known_names = data.get("names", [])
                    self.known_encodings = data.get("encodings", [])
                    self.known_metadata = data.get("metadata", [])
                print(f"[FaceDatabase] Loaded {len(self.known_names)} known faces.")
            except Exception as e:
                print(f"[FaceDatabase] Error loading database: {e}")

    def save_database(self):
        os.makedirs(os.path.dirname(KNOWN_FACES_FILE), exist_ok=True)
        data = {
            "names": self.known_names,
            "encodings": self.known_encodings,
            "metadata": self.known_metadata
        }
        with open(KNOWN_FACES_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"[FaceDatabase] Saved {len(self.known_names)} known faces.")

    def add_face(self, name, encoding, metadata=None):
        if encoding is None:
            return False
        idx = self._find_name_index(name)
        if idx >= 0:
            self.known_encodings[idx].append(encoding)
            if metadata:
                self.known_metadata[idx].append(metadata)
        else:
            self.known_names.append(name)
            self.known_encodings.append([encoding])
            self.known_metadata.append([metadata] if metadata else [])
        self.save_database()
        return True

    def remove_face(self, name):
        idx = self._find_name_index(name)
        if idx >= 0:
            self.known_names.pop(idx)
            self.known_encodings.pop(idx)
            self.known_metadata.pop(idx)
            self.save_database()
            return True
        return False

    def _find_name_index(self, name):
        for i, n in enumerate(self.known_names):
            if n.lower() == name.lower():
                return i
        return -1

    def recognize_face(self, encoding, tolerance=None):
        if encoding is None or len(self.known_encodings) == 0:
            return "Unknown", 0.0
        if tolerance is None:
            tolerance = FACE_RECOGNITION_TOLERANCE

        best_match = "Unknown"
        best_distance = 1.0

        for i, enc_list in enumerate(self.known_encodings):
            for known_enc in enc_list:
                dist = np.linalg.norm(encoding - known_enc)
                if dist < best_distance:
                    best_distance = dist
                    if dist < tolerance:
                        best_match = self.known_names[i]

        confidence = max(0, 1.0 - best_distance)
        return best_match, confidence

    def get_all_names(self):
        return list(self.known_names)

    def get_face_count(self, name=None):
        if name:
            idx = self._find_name_index(name)
            if idx >= 0:
                return len(self.known_encodings[idx])
            return 0
        return sum(len(enc) for enc in self.known_encodings)


class FaceRecognizer:
    def __init__(self):
        self.encoder = FaceEncoder()
        self.database = FaceDatabase()
        self.ml_recognizer = MLFaceRecognizer()
        self.initialized = self.encoder.initialized or self.ml_recognizer.initialized

        if self.ml_recognizer.initialized:
            print("[FaceRecognizer] Using ML-trained model for recognition.")
        elif self.encoder.initialized:
            print("[FaceRecognizer] Using OpenFace encoder for recognition.")

    def recognize_frame(self, frame, faces):
        if not self.initialized:
            return [], []

        if self.ml_recognizer.initialized:
            results = self.ml_recognizer.recognize_faces(frame, faces)
            labels = []
            for r in results:
                if r["name"] != "Unknown":
                    labels.append(f"{r['name']} ({r['confidence']:.0%})")
                else:
                    labels.append("Unknown")
            return [], labels

        encodings = self.encoder.encode_faces_in_frame(frame, faces)
        labels = []
        for enc in encodings:
            if enc is not None:
                name, conf = self.database.recognize_face(enc)
                if name != "Unknown":
                    labels.append(f"{name} ({conf:.0%})")
                else:
                    labels.append("Unknown")
            else:
                labels.append("Unknown")
        return encodings, labels

    def add_person(self, name, frame, face_bbox):
        encoding = self.encoder.encode_face(frame, face_bbox)
        if encoding is not None:
            timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(CAPTURED_DIR, f"{name}_{timestamp}.jpg")
            x1, y1, x2, y2 = face_bbox
            cv2.imwrite(img_path, frame[y1:y2, x1:x2])
            metadata = {"image_path": img_path, "timestamp": timestamp}
            self.database.add_face(name, encoding, metadata)
            return True, f"Added {name} to database."
        return False, "Failed to encode face."

    def save_capture(self, frame, name="capture"):
        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CAPTURED_DIR, f"{name}_{timestamp}.jpg")
        cv2.imwrite(path, frame)
        return path
