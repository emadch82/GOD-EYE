import cv2
import numpy as np
import os
import pickle
from datetime import datetime
from config import (
    FACE_DB_DIR, CAPTURED_DIR, KNOWN_FACES_FILE,
    FACE_DETECT_PROTOTXT, FACE_DETECT_MODEL, FACE_ENCODING_MODEL
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


class FaceTrainer:
    def __init__(self):
        self.detect_net = None
        self.encoding_net = None
        self.initialized = False
        self._load_models()

    def _load_models(self):
        try:
            if os.path.exists(FACE_DETECT_PROTOTXT) and os.path.exists(FACE_DETECT_MODEL):
                self.detect_net = cv2.dnn.readNetFromCaffe(FACE_DETECT_PROTOTXT, FACE_DETECT_MODEL)
            if os.path.exists(FACE_ENCODING_MODEL):
                self.encoding_net = cv2.dnn.readNetFromTorch(FACE_ENCODING_MODEL)
            if self.detect_net and self.encoding_net:
                self.initialized = True
                print("[FaceTrainer] Models loaded.")
        except Exception as e:
            print(f"[FaceTrainer] Error: {e}")

    def prepare_dataset(self, dataset_dir):
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory not found: {dataset_dir}")
            return None, None

        encodings = []
        labels = []
        person_dirs = [d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d))]

        if not person_dirs:
            print("No person directories found.")
            return None, None

        print(f"Found {len(person_dirs)} people in dataset.")

        for person_name in person_dirs:
            person_path = os.path.join(dataset_dir, person_name)
            image_files = [f for f in os.listdir(person_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            print(f"  Processing {person_name}: {len(image_files)} images")
            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                faces = self._detect_faces(img)
                for face in faces:
                    encoding = self._encode_face(img, face)
                    if encoding is not None:
                        encodings.append(encoding)
                        labels.append(person_name)

        print(f"\nTotal encodings: {len(encodings)}")
        print(f"Total labels: {len(set(labels))}")
        return encodings, labels

    def _detect_faces(self, frame):
        if self.detect_net is None:
            return []
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        self.detect_net.setInput(blob)
        detections = self.detect_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                faces.append((x1, y1, x2, y2))
        return faces

    def _encode_face(self, frame, bbox):
        if self.encoding_net is None:
            return None
        x1, y1, x2, y2 = bbox
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        face = cv2.resize(face, (96, 96))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        self.encoding_net.setInput(blob)
        encoding = self.encoding_net.forward().flatten()
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding = encoding / norm
        return encoding

    def train_from_directory(self, dataset_dir, output_dir=None):
        if not self.initialized:
            print("Models not loaded.")
            return False

        print("=" * 60)
        print("  FACE RECOGNITION TRAINING")
        print("=" * 60)

        encodings, labels = self.prepare_dataset(dataset_dir)
        if encodings is None or len(encodings) == 0:
            print("No training data found.")
            return False

        print(f"\nTraining with {len(encodings)} encodings from {len(set(labels))} people.")

        known_names = []
        known_encodings = []

        for name in set(labels):
            name_encs = [enc for enc, lbl in zip(encodings, labels) if lbl == name]
            known_names.append(name)
            known_encodings.append(name_encs)
            print(f"  {name}: {len(name_encs)} encodings")

        data = {
            "names": known_names,
            "encodings": known_encodings,
            "metadata": [{"trained": True, "timestamp": datetime.now().isoformat()} for _ in known_names]
        }

        if output_dir is None:
            output_dir = FACE_DB_DIR
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "known_faces.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(data, f)

        print(f"\nModel saved to: {output_file}")
        print(f"Total people: {len(known_names)}")
        print("Training complete!")
        return True

    def evaluate(self, test_dataset_dir):
        if not self.initialized:
            return

        print("Evaluating on test dataset...")
        encodings, labels = self.prepare_dataset(test_dataset_dir)
        if encodings is None:
            return

        predictions = []
        true_labels = []

        known_data = self._load_known_faces()
        if not known_data:
            print("No known faces for comparison.")
            return

        for enc, true_label in zip(encodings, labels):
            best_name = "Unknown"
            best_dist = 1.0
            for name, name_encs in zip(known_data["names"], known_data["encodings"]):
                for known_enc in name_encs:
                    dist = np.linalg.norm(enc - known_enc)
                    if dist < best_dist:
                        best_dist = dist
                        if dist < 0.6:
                            best_name = name
            predictions.append(best_name)
            true_labels.append(true_label)

        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        print(f"Accuracy: {accuracy_score(true_labels, predictions):.2%}")

    def _load_known_faces(self):
        if os.path.exists(KNOWN_FACES_FILE):
            with open(KNOWN_FACES_FILE, "rb") as f:
                return pickle.load(f)
        return None


def capture_training_data(output_dir, camera_index=0, num_photos=20):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    print(f"Capturing {num_photos} photos...")
    count = 0
    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        cv2.imshow("Capture Training Data", frame)
        cv2.waitKey(500)

        path = os.path.join(output_dir, f"photo_{count+1:04d}.jpg")
        cv2.imwrite(path, frame)
        count += 1
        print(f"  Captured {count}/{num_photos}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Photos saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    trainer = FaceTrainer()
    if len(sys.argv) > 1:
        if sys.argv[1] == "train" and len(sys.argv) > 2:
            trainer.train_from_directory(sys.argv[2])
        elif sys.argv[1] == "evaluate" and len(sys.argv) > 2:
            trainer.evaluate(sys.argv[2])
        elif sys.argv[1] == "capture" and len(sys.argv) > 2:
            name = sys.argv[2]
            num = int(sys.argv[3]) if len(sys.argv) > 3 else 20
            output = os.path.join(CAPTURED_DIR, name)
            capture_training_data(output, num_photos=num)
    else:
        print("Usage:")
        print("  python trainer.py train DATASET_DIR")
        print("  python trainer.py evaluate TEST_DIR")
        print("  python trainer.py capture NAME [NUM_PHOTOS]")
