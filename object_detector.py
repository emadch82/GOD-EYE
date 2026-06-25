import cv2
import numpy as np
from ultralytics import YOLO
from config import YOLO_MODEL, YOLO_CONFIDENCE, COLORS


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

CATEGORY_COLORS = {
    "person": COLORS["green"],
    "vehicle": COLORS["blue"],
    "animal": COLORS["yellow"],
    "object": COLORS["cyan"],
    "food": COLORS["red"],
    "furniture": (128, 0, 128),
    "electronics": (255, 128, 0),
}

ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
VEHICLE_CLASSES = {"car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "bicycle"}
FOOD_CLASSES = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
FURNITURE_CLASSES = {"chair", "couch", "bed", "dining table", "potted plant"}
ELECTRONICS_CLASSES = {"tv", "laptop", "mouse", "remote", "keyboard", "cell phone"}


def get_category(class_name):
    if class_name == "person":
        return "person"
    if class_name in ANIMAL_CLASSES:
        return "animal"
    if class_name in VEHICLE_CLASSES:
        return "vehicle"
    if class_name in FOOD_CLASSES:
        return "food"
    if class_name in FURNITURE_CLASSES:
        return "furniture"
    if class_name in ELECTRONICS_CLASSES:
        return "electronics"
    return "object"


def get_category_color(category):
    return CATEGORY_COLORS.get(category, COLORS["object"])


class ObjectDetector:
    def __init__(self, model_name=None, confidence=None):
        self.model = None
        self.initialized = False
        self.model_name = model_name or YOLO_MODEL
        self.confidence = confidence or YOLO_CONFIDENCE
        self._load_model()

    def _load_model(self):
        try:
            print(f"[ObjectDetector] Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)
            self.initialized = True
            print("[ObjectDetector] YOLO model loaded successfully.")
            return True
        except Exception as e:
            print(f"[ObjectDetector] Error loading YOLO model: {e}")
            return False

    def detect(self, frame, confidence=None):
        if not self.initialized:
            return []
        if confidence is None:
            confidence = self.confidence

        results = self.model(frame, conf=confidence, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    category = get_category(class_name)

                    detections.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": class_name,
                        "category": category,
                    })
        return detections

    def draw_detections(self, frame, detections, show_confidence=True):
        result = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            class_name = det["class_name"]
            category = det["category"]
            color = get_category_color(category)

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            label = class_name
            if show_confidence:
                label = f"{class_name}: {conf:.0%}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(result, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 1, cv2.LINE_AA)

        return result

    def get_detection_summary(self, detections):
        summary = {}
        for det in detections:
            name = det["class_name"]
            if name in summary:
                summary[name]["count"] += 1
                summary[name]["avg_conf"] = (summary[name]["avg_conf"] + det["confidence"]) / 2
            else:
                summary[name] = {
                    "count": 1,
                    "category": det["category"],
                    "avg_conf": det["confidence"]
                }
        return summary

    def detect_and_classify(self, frame):
        detections = self.detect(frame)
        summary = self.get_detection_summary(detections)
        categories_found = set(d["category"] for d in detections)
        return detections, summary, categories_found

    def filter_by_category(self, detections, category):
        return [d for d in detections if d["category"] == category]

    def filter_by_class(self, detections, class_name):
        return [d for d in detections if d["class_name"].lower() == class_name.lower()]

    def filter_by_confidence(self, detections, min_confidence):
        return [d for d in detections if d["confidence"] >= min_confidence]
