import cv2
import numpy as np
import os, sys, pickle, time, threading, collections
import customtkinter as ctk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FACE_DETECT_PROTOTXT, FACE_DETECT_MODEL, FACE_DB_DIR,
    KNOWN_FACES_FILE, CAPTURED_DIR, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    FACE_DETECT_CONFIDENCE, FACE_ENCODING_MODEL
)
from face_detector import FaceDetector

try:
    import ai_core
    RUST = True
except ImportError:
    RUST = False

ONNX_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.onnx")
PROFILES_FILE = os.path.join(FACE_DB_DIR, "profiles.pkl")

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ObjDet:
    def __init__(self):
        self.net = None
        self.frame_count = 0
        self.last_results = []
        self._load()

    def _load(self):
        if not os.path.exists(ONNX_MODEL):
            return
        try:
            self.net = cv2.dnn.readNetFromONNX(ONNX_MODEL)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except:
            pass

    def detect(self, frame, skip=5):
        self.frame_count += 1
        if self.frame_count % skip != 0:
            return self.last_results
        if self.net is None:
            return []
        h, w = frame.shape[:2]
        inp = cv2.resize(frame, (320, 320))
        blob = cv2.dnn.blobFromImage(inp, 1.0 / 255.0, (320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        detections = []
        sx, sy = w / 320.0, h / 320.0
        for i in range(out.shape[2]):
            scores = out[0, 4:, i, 0]
            cls = int(np.argmax(scores))
            conf = float(scores[cls])
            if conf < 0.3:
                continue
            cx, cy, bw, bh = out[0, 0, i, 0], out[0, 1, i, 0], out[0, 2, i, 0], out[0, 3, i, 0]
            x1 = max(0, int((cx - bw / 2) * sx))
            y1 = max(0, int((cy - bh / 2) * sy))
            x2 = min(w, int((cx + bw / 2) * sx))
            y2 = min(h, int((cy + bh / 2) * sy))
            if x2 - x1 > 10 and y2 - y1 > 10:
                name = COCO_NAMES[cls] if cls < len(COCO_NAMES) else "object"
                detections.append({"bbox": (x1, y1, x2, y2), "name": name, "confidence": conf})
        self.last_results = detections
        return detections


SFACE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sface_2021dec.onnx")


class FaceEncoder:
    def __init__(self):
        self.recognizer = None
        self.initialized = False
        self._load()

    def _load(self):
        if os.path.exists(SFACE_MODEL):
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
                self.initialized = True
                print("[FaceEncoder] SFace loaded (99.4% accuracy)")
                return
            except Exception as e:
                print(f"[FaceEncoder] SFace failed: {e}")

        if os.path.exists(FACE_ENCODING_MODEL):
            try:
                self.net = cv2.dnn.readNetFromTorch(FACE_ENCODING_MODEL)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.initialized = True
                print("[FaceEncoder] OpenFace fallback loaded")
            except:
                pass

    def encode(self, frame, x1, y1, x2, y2, landmarks=None):
        if not self.initialized:
            return None
        h, w = frame.shape[:2]

        if hasattr(self, 'recognizer') and self.recognizer is not None:
            pad_w = int((x2 - x1) * 0.15)
            pad_h = int((y2 - y1) * 0.15)
            cx1 = max(0, x1 - pad_w)
            cy1 = max(0, y1 - pad_h)
            cx2 = min(w, x2 + pad_w)
            cy2 = min(h, y2 + pad_h)
            if cx2 - cx1 < 30 or cy2 - cy1 < 30:
                return None
            face_roi = frame[cy1:cy2, cx1:cx2]

            if landmarks:
                aligned = self._align_face(frame, landmarks, 112)
                if aligned is not None:
                    face_roi = aligned

            result = self.recognizer.feature(face_roi)
            if result is not None:
                return result.flatten().astype(np.float32)
            return None

        pad_w = int((x2 - x1) * 0.2)
        pad_h = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        if x2 - x1 < 30 or y2 - y1 < 30:
            return None
        face_roi = frame[y1:y2, x1:x2]
        blob = cv2.dnn.blobFromImage(face_roi, 1.0 / 255.0, (96, 96), swapRB=True, crop=False)
        self.net.setInput(blob)
        encoding = self.net.forward().flatten()
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding = encoding / norm
        return encoding

    def _align_face(self, frame, landmarks, size=112):
        if len(landmarks) < 5:
            return None
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]

        desired_left_eye = (0.35, 0.35)
        desired_right_eye = (0.65, 0.35)

        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = desired_right_eye[0] - desired_left_eye[0]
        desired_dist *= size
        scale = desired_dist / dist

        eyes_center = ((left_eye[0] + right_eye[0]) / 2.0,
                       (left_eye[1] + right_eye[1]) / 2.0)

        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        M[0, 2] += (size * 0.5) - eyes_center[0]
        M[1, 2] += (size * desired_left_eye[1]) - eyes_center[1]

        h, w = frame.shape[:2]
        aligned = cv2.warpAffine(frame, M, (size, size),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return aligned

    def encode_multi(self, frame, x1, y1, x2, y2, landmarks=None, count=5):
        encodings = []
        h, w = frame.shape[:2]
        fw, fh = x2 - x1, y2 - y1
        for i in range(count):
            px = int(fw * 0.04 * ((i % 3) - 1))
            py = int(fh * 0.04 * (((i // 3) % 3) - 1))
            nx1 = max(0, x1 + px)
            ny1 = max(0, y1 + py)
            nx2 = min(w, x2 + px)
            ny2 = min(h, y2 + py)
            enc = self.encode(frame, nx1, ny1, nx2, ny2, landmarks)
            if enc is not None:
                encodings.append(enc)
        if not encodings:
            return None
        avg = np.mean(encodings, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg

    def match(self, enc1, enc2):
        if enc1 is None or enc2 is None:
            return 0.0
        if hasattr(self, 'recognizer') and self.recognizer is not None:
            return self.recognizer.match(enc1, enc2)
        n1 = np.linalg.norm(enc1)
        n2 = np.linalg.norm(enc2)
        if n1 > 0 and n2 > 0:
            return float(np.dot(enc1 / n1, enc2 / n2))
        return 0.0


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        tag = "RUST" if RUST else "PYTHON"
        self.title("AI Vision Pro")
        self.geometry("1920x1080")
        self.configure(fg_color="#0d1117")
        self.state("zoomed")

        self.webcam = None
        self.face_det = None
        self.face_enc = None
        self.obj_det = None
        self.running = False
        self.paused = False
        self.known_encodings = []
        self.known_names = []
        self.profiles = {}
        self.fps_q = collections.deque(maxlen=30)
        self.ready = False
        self.photo = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.cam_w = FRAME_WIDTH
        self.cam_h = FRAME_HEIGHT
        self.detect_faces = True
        self.detect_objects = True
        self._enc_counter = 0
        self._enc_cache = {}
        self.last_face_key = ""
        self.info_labels = []

        self._load_data()
        self._build()
        self.after(100, self._load_bg)
        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.mainloop()

    def _load_data(self):
        if os.path.exists(KNOWN_FACES_FILE):
            try:
                with open(KNOWN_FACES_FILE, "rb") as f:
                    d = pickle.load(f)
                self.known_names = d.get("names", [])
                self.known_encodings = d.get("encodings", [])
            except:
                pass
        if os.path.exists(PROFILES_FILE):
            try:
                with open(PROFILES_FILE, "rb") as f:
                    self.profiles = pickle.load(f)
            except:
                pass

    def _save_data(self):
        os.makedirs(FACE_DB_DIR, exist_ok=True)
        with open(KNOWN_FACES_FILE, "wb") as f:
            pickle.dump({"names": self.known_names, "encodings": self.known_encodings, "metadata": []}, f)
        with open(PROFILES_FILE, "wb") as f:
            pickle.dump(self.profiles, f)

    def _build(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color="#161b22")
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=20, pady=(20, 5))

        ctk.CTkLabel(logo_frame, text="AI VISION", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#58a6ff").pack(side="left")
        ctk.CTkLabel(logo_frame, text=" PRO", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#8b949e").pack(side="left")

        engine = "RUST" if RUST else "PYTHON"
        self.status_badge = ctk.CTkLabel(sidebar, text=f"Engine: {engine}",
                                         font=ctk.CTkFont(size=11),
                                         text_color="#8b949e", fg_color="#21262d",
                                         corner_radius=6, padx=8, pady=3)
        self.status_badge.pack(padx=20, anchor="w", pady=(0, 15))

        ctk.CTkFrame(sidebar, height=1, fg_color="#30363d").pack(fill="x", padx=20, pady=(0, 15))

        ctk.CTkLabel(sidebar, text="CAMERA", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#8b949e").pack(padx=20, anchor="w", pady=(0, 5))

        self.btn_start = ctk.CTkButton(sidebar, text="Start Camera", height=40,
                                       fg_color="#238636", hover_color="#2ea043",
                                       font=ctk.CTkFont(size=13, weight="bold"),
                                       corner_radius=8, command=self._start)
        self.btn_start.pack(padx=15, pady=3, fill="x")

        btn_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_row.pack(padx=15, pady=3, fill="x")

        self.btn_stop = ctk.CTkButton(btn_row, text="Stop", height=36, width=100,
                                      fg_color="#da3633", hover_color="#f85149",
                                      font=ctk.CTkFont(size=12), corner_radius=8,
                                      command=self._stop)
        self.btn_stop.pack(side="left", padx=(0, 4), expand=True, fill="x")

        self.btn_pause = ctk.CTkButton(btn_row, text="Pause", height=36, width=100,
                                       fg_color="#9e6a03", hover_color="#d29922",
                                       font=ctk.CTkFont(size=12), corner_radius=8,
                                       command=self._pause)
        self.btn_pause.pack(side="right", padx=(4, 0), expand=True, fill="x")

        ctk.CTkFrame(sidebar, height=1, fg_color="#30363d").pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(sidebar, text="DETECT", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#8b949e").pack(padx=20, anchor="w", pady=(0, 5))

        self.face_var = ctk.BooleanVar(value=True)
        self.obj_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(sidebar, text="Face Detection", variable=self.face_var,
                      font=ctk.CTkFont(size=12), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#58a6ff").pack(padx=20, anchor="w", pady=2)
        ctk.CTkSwitch(sidebar, text="Object Detection", variable=self.obj_var,
                      font=ctk.CTkFont(size=12), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#58a6ff").pack(padx=20, anchor="w", pady=2)

        ctk.CTkFrame(sidebar, height=1, fg_color="#30363d").pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(sidebar, text="ACTIONS", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#8b949e").pack(padx=20, anchor="w", pady=(0, 5))

        ctk.CTkButton(sidebar, text="Add Person", height=36,
                      fg_color="#1f6feb", hover_color="#388bfd",
                      font=ctk.CTkFont(size=12), corner_radius=8,
                      command=self._add_face).pack(padx=15, pady=2, fill="x")

        ctk.CTkButton(sidebar, text="Remove Person", height=36,
                      fg_color="#8957e5", hover_color="#a371f7",
                      font=ctk.CTkFont(size=12), corner_radius=8,
                      command=self._remove_face).pack(padx=15, pady=2, fill="x")

        ctk.CTkButton(sidebar, text="List People", height=36,
                      fg_color="#30363d", hover_color="#484f58",
                      text_color="#c9d1d9",
                      font=ctk.CTkFont(size=12), corner_radius=8,
                      command=self._list_faces).pack(padx=15, pady=2, fill="x")

        ctk.CTkFrame(sidebar, height=1, fg_color="#30363d").pack(fill="x", padx=20, pady=10)

        ctk.CTkButton(sidebar, text="Screenshot", height=36,
                      fg_color="#30363d", hover_color="#484f58",
                      text_color="#c9d1d9",
                      font=ctk.CTkFont(size=12), corner_radius=8,
                      command=self._screenshot).pack(padx=15, pady=2, fill="x")

        ctk.CTkButton(sidebar, text="Open Image", height=36,
                      fg_color="#30363d", hover_color="#484f58",
                      text_color="#c9d1d9",
                      font=ctk.CTkFont(size=12), corner_radius=8,
                      command=self._open_image).pack(padx=15, pady=2, fill="x")

        ctk.CTkButton(sidebar, text="Quit", height=36,
                      fg_color="#da3633", hover_color="#f85149",
                      font=ctk.CTkFont(size=12), corner_radius=8,
                      command=self._quit).pack(padx=15, pady=(10, 15), fill="x", side="bottom")

        main = ctk.CTkFrame(self, fg_color="#0d1117", corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        self.canvas_label = ctk.CTkLabel(main, text="", fg_color="#010409",
                                         corner_radius=12, anchor="center")
        self.canvas_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))

        self.info_panel = ctk.CTkFrame(main, fg_color="#161b22", corner_radius=10, width=320)
        self.info_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=(8, 4))
        self.info_panel.grid_propagate(False)

        ctk.CTkLabel(self.info_panel, text="DETECTED FACES",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#58a6ff").pack(padx=15, pady=(12, 8), anchor="w")

        self.info_scroll = ctk.CTkScrollableFrame(self.info_panel, fg_color="transparent",
                                                   scrollbar_button_color="#30363d",
                                                   scrollbar_button_hover_color="#484f58")
        self.info_scroll.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        bottom = ctk.CTkFrame(main, fg_color="#161b22", corner_radius=10, height=55)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
        bottom.grid_propagate(False)

        self.stat_fps = ctk.CTkLabel(bottom, text="FPS: --",
                                     font=ctk.CTkFont(size=14, weight="bold"),
                                     text_color="#3fb950", fg_color="#0d1117",
                                     corner_radius=8, padx=12, pady=6)
        self.stat_fps.pack(side="left", padx=(12, 6), pady=10)

        self.stat_faces = ctk.CTkLabel(bottom, text="Faces: 0",
                                       font=ctk.CTkFont(size=14, weight="bold"),
                                       text_color="#d29922", fg_color="#0d1117",
                                       corner_radius=8, padx=12, pady=6)
        self.stat_faces.pack(side="left", padx=6, pady=10)

        self.stat_objects = ctk.CTkLabel(bottom, text="Objects: 0",
                                         font=ctk.CTkFont(size=14, weight="bold"),
                                         text_color="#58a6ff", fg_color="#0d1117",
                                         corner_radius=8, padx=12, pady=6)
        self.stat_objects.pack(side="left", padx=6, pady=10)

        self.s_info = ctk.CTkLabel(bottom, text="",
                                   font=ctk.CTkFont(size=11),
                                   text_color="#8b949e")
        self.s_info.pack(side="left", padx=15, expand=True, fill="x")

        self.s_time = ctk.CTkLabel(bottom, text="",
                                   font=ctk.CTkFont(size=11),
                                   text_color="#8b949e")
        self.s_time.pack(side="right", padx=12)

    def _load_bg(self):
        def load():
            try:
                self.face_det = FaceDetector()
            except Exception as e:
                print("Face detector error:", e)
            try:
                self.face_enc = FaceEncoder()
            except Exception as e:
                print("Face encoder error:", e)
            try:
                self.obj_det = ObjDet()
            except Exception as e:
                print("Object detector error:", e)
            self.ready = True
            self.after(0, lambda: self.status_badge.configure(
                text=f"Ready [{('RUST' if RUST else 'PYTHON')}]",
                fg_color="#238636", text_color="#ffffff"))
        threading.Thread(target=load, daemon=True).start()

    def _start(self):
        if self.running:
            return
        if not self.ready:
            messagebox.showinfo("Wait", "Models still loading...")
            return
        self.webcam = cv2.VideoCapture(CAMERA_INDEX)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.webcam.isOpened():
            messagebox.showerror("Error", "Camera not available!")
            return
        self.running = True
        self.paused = False
        self.status_badge.configure(text="REC", fg_color="#da3633", text_color="#ffffff")
        threading.Thread(target=self._reader, daemon=True).start()
        self._loop()

    def _reader(self):
        while self.running:
            try:
                if self.webcam and self.webcam.isOpened():
                    ret, frame = self.webcam.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        with self.lock:
                            self.latest_frame = frame.copy()
            except:
                pass
            time.sleep(0.001)

    def _stop(self):
        self.running = False
        time.sleep(0.02)
        if self.webcam:
            self.webcam.release()
            self.webcam = None
        self.status_badge.configure(text="STOPPED", fg_color="#da3633", text_color="#ffffff")
        self.canvas_label.configure(image=None, text="Camera Off")
        self.stat_fps.configure(text="FPS: --")
        self.stat_faces.configure(text="Faces: 0")
        self.stat_objects.configure(text="Objects: 0")

    def _pause(self):
        self.paused = not self.paused
        if self.paused:
            self.status_badge.configure(text="PAUSED", fg_color="#9e6a03", text_color="#ffffff")
        else:
            self.status_badge.configure(text="REC", fg_color="#da3633", text_color="#ffffff")

    def _match_face(self, encoding):
        if not self.known_encodings or encoding is None:
            return "Unknown", 0.0

        encoding = np.array(encoding, dtype=np.float32)
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding = encoding / norm

        best_name = "Unknown"
        best_score = 0.0
        best_idx = -1

        for i, known_enc in enumerate(self.known_encodings):
            if known_enc is None or not isinstance(known_enc, np.ndarray):
                continue
            if len(known_enc) != len(encoding):
                continue

            if self.face_enc and hasattr(self.face_enc, 'recognizer') and self.face_enc.recognizer is not None:
                score = self.face_enc.recognizer.match(encoding, known_enc)
            else:
                kn = np.linalg.norm(known_enc)
                if kn > 0:
                    score = float(np.dot(encoding, known_enc / kn))
                else:
                    continue

            if score > best_score:
                best_score = score
                best_name = self.known_names[i] if i < len(self.known_names) else "Unknown"
                best_idx = i

        if best_score > 0.45:
            if best_idx >= 0:
                old = self.known_encodings[best_idx]
                if old is not None and isinstance(old, np.ndarray) and len(old) == len(encoding):
                    self.known_encodings[best_idx] = (old * 0.85 + encoding * 0.15).astype(np.float32)
            return best_name, best_score
        return "Unknown", best_score

    def _loop(self):
        if not self.running:
            return
        if self.paused:
            self.after(10, self._loop)
            return

        with self.lock:
            frame = self.latest_frame

        if frame is None:
            self.after(3, self._loop)
            return

        t0 = time.time()
        faces = []
        objects = []

        try:
            if self.detect_faces and self.face_det and self.face_det.initialized:
                faces = self.face_det.detect(frame)
            if self.detect_objects and self.obj_det:
                objects = self.obj_det.detect(frame, skip=5)
        except:
            pass

        face_infos = []
        self._enc_counter += 1

        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (129, 185, 16), 2)
            cl = 12
            cv2.line(frame, (x1, y1), (x1 + cl, y1), (129, 185, 16), 3)
            cv2.line(frame, (x1, y1), (x1, y1 + cl), (129, 185, 16), 3)
            cv2.line(frame, (x2, y1), (x2 - cl, y1), (129, 185, 16), 3)
            cv2.line(frame, (x2, y1), (x2, y1 + cl), (129, 185, 16), 3)
            cv2.line(frame, (x1, y2), (x1 + cl, y2), (129, 185, 16), 3)
            cv2.line(frame, (x1, y2), (x1, y2 - cl), (129, 185, 16), 3)
            cv2.line(frame, (x2, y2), (x2 - cl, y2), (129, 185, 16), 3)
            cv2.line(frame, (x2, y2), (x2, y2 - cl), (129, 185, 16), 3)

            name = "Unknown"
            profile_lines = []
            face_conf = f.get("confidence", 0)

            if face_conf > 0.7 and self.face_enc and self.face_enc.initialized:
                landmarks = f.get("landmarks", [])
                cache_key = "{}_{}_{}_{}".format(x1 // 20, y1 // 20, x2 // 20, y2 // 20)
                if self._enc_counter % 3 == 0 or cache_key not in self._enc_cache:
                    enc = self.face_enc.encode(frame, x1, y1, x2, y2, landmarks)
                    if enc is not None:
                        self._enc_cache[cache_key] = enc
                        keys = list(self._enc_cache.keys())
                        if len(keys) > 10:
                            for k in keys[:-10]:
                                del self._enc_cache[k]
                else:
                    enc = self._enc_cache.get(cache_key)
                if enc is not None:
                    matched_name, dist = self._match_face(enc)
                    name = matched_name
                    if name != "Unknown" and name in self.profiles:
                        p = self.profiles[name]
                        if p.get("name"):
                            profile_lines.append(p["name"])
                        if p.get("age"):
                            profile_lines.append("Age: {}".format(p["age"]))
                        if p.get("job"):
                            profile_lines.append(p["job"])
                        if p.get("desc"):
                            profile_lines.append(p["desc"])

            is_known = name != "Unknown"
            face_infos.append({"name": name, "lines": profile_lines, "known": is_known})

            status = "KNOWN" if is_known else "UNKNOWN"
            sc = (129, 185, 16) if is_known else (68, 68, 239)
            cv2.putText(frame, status, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1, cv2.LINE_AA)

        for o in objects:
            x1, y1, x2, y2 = o["bbox"]
            color = (6, 182, 212) if o["name"] in ["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"] else (139, 92, 246)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = "{} {:.0%}".format(o["name"], o["confidence"])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        dt = time.time() - t0
        if dt > 0:
            self.fps_q.append(1.0 / dt)
        fps = sum(self.fps_q) / len(self.fps_q) if self.fps_q else 0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        cw = max(self.canvas_label.winfo_width(), 200)
        ch = max(self.canvas_label.winfo_height(), 200)
        s = min(cw / pil.width, ch / pil.height)
        pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas_label.configure(image=self.photo, text="")

        self.after(0, lambda: self._update_stats(fps, len(faces), len(objects)))

        face_key = str([(f["name"], f["known"], i) for i, f in enumerate(face_infos)])
        if face_key != self.last_face_key:
            self.last_face_key = face_key
            self.after(0, lambda: self._update_info_panel(face_infos))

        self.after(15, self._loop)

    def _update_stats(self, fps, face_count, obj_count):
        self.stat_fps.configure(text="FPS: {:.0f}".format(fps))
        self.stat_faces.configure(text="Faces: {}".format(face_count))
        self.stat_objects.configure(text="Objects: {}".format(obj_count))
        self.s_time.configure(text=datetime.now().strftime("%H:%M:%S"))
        self.s_info.configure(text="{}x{} | Engine: {}".format(
            self.cam_w, self.cam_h, "RUST" if RUST else "PYTHON"))

    def _update_info_panel(self, face_infos):
        for widget in self.info_scroll.winfo_children():
            widget.destroy()

        if not face_infos:
            ctk.CTkLabel(self.info_scroll, text="No faces detected",
                         font=ctk.CTkFont(size=11), text_color="#484f58").pack(pady=10)
            return

        for info in face_infos:
            is_known = info["known"]
            border_color = "#238636" if is_known else "#da3633"

            card = ctk.CTkFrame(self.info_scroll, fg_color="#0d1117",
                                corner_radius=8, border_width=1,
                                border_color=border_color)
            card.pack(fill="x", pady=3, padx=2)

            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=10, pady=(8, 2))

            status = "KNOWN" if is_known else "UNKNOWN"
            status_color = "#3fb950" if is_known else "#f85149"

            ctk.CTkLabel(top, text=status, font=ctk.CTkFont(size=9, weight="bold"),
                         text_color=status_color).pack(side="right")

            ctk.CTkLabel(top, text=info["name"],
                         font=ctk.CTkFont(size=14, weight="bold"),
                         text_color="#e6edf3").pack(side="left")

            if info["lines"]:
                sep = ctk.CTkFrame(card, height=1, fg_color="#21262d")
                sep.pack(fill="x", padx=10, pady=(4, 4))
                for line in info["lines"]:
                    ctk.CTkLabel(card, text="  {}".format(line),
                                 font=ctk.CTkFont(size=11),
                                 text_color="#8b949e", anchor="w").pack(fill="x", padx=10, pady=1)

            ctk.CTkFrame(card, height=4, fg_color="transparent").pack()

    def _add_face(self):
        if not self.running:
            messagebox.showinfo("Info", "Start camera first!")
            return

        win = ctk.CTkToplevel(self)
        win.title("Add Person")
        win.geometry("400x520")
        win.configure(fg_color="#0d1117")
        win.resizable(False, False)
        win.grab_set()

        ctk.CTkLabel(win, text="ADD NEW PERSON",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#58a6ff").pack(pady=(20, 15))

        fields = {}
        for key, label, placeholder in [
            ("name", "Full Name *", "Enter full name..."),
            ("age", "Age", "e.g. 25"),
            ("job", "Job / Title", "e.g. Software Engineer"),
            ("desc", "Description", "e.g. Project manager"),
        ]:
            ctk.CTkLabel(win, text=label, font=ctk.CTkFont(size=11, weight="bold"),
                         text_color="#8b949e", anchor="w").pack(fill="x", padx=30, pady=(8, 2))
            entry = ctk.CTkEntry(win, placeholder_text=placeholder, height=38,
                                 fg_color="#161b22", border_color="#30363d",
                                 text_color="#e6edf3", font=ctk.CTkFont(size=12),
                                 corner_radius=8)
            entry.pack(fill="x", padx=30, pady=(0, 2))
            fields[key] = entry

        def get_val(key):
            return fields[key].get().strip()

        def on_submit():
            name = get_val("name")
            if not name:
                messagebox.showwarning("Warning", "Name is required!", parent=win)
                return
            messagebox.showinfo("Capture", "Look at the camera and hold still...", parent=win)

            encodings = []
            best_conf = 0
            best_frame = None
            best_face = None

            for _ in range(30):
                time.sleep(0.1)
                with self.lock:
                    frame = self.latest_frame
                if frame is None:
                    continue
                try:
                    f = self.face_det.detect(frame) if self.face_det else []
                    if f:
                        best_f = max(f, key=lambda x: x["confidence"])
                        if best_f["confidence"] > best_conf:
                            best_conf = best_f["confidence"]
                            best_face = best_f
                            best_frame = frame.copy()
                        if self.face_enc and self.face_enc.initialized:
                            x1, y1, x2, y2 = best_f["bbox"]
                            lm = best_f.get("landmarks", [])
                            enc = self.face_enc.encode_multi(frame, x1, y1, x2, y2, lm, count=5)
                            if enc is not None:
                                encodings.append(enc)
                    if best_conf > 0.7 and len(encodings) >= 5:
                        break
                except:
                    pass

            if best_face is None:
                messagebox.showinfo("Info", "No face detected! Try again.", parent=win)
                return

            avg_enc = None
            if encodings:
                avg_enc = np.mean(encodings, axis=0).astype(np.float32)
                norm = np.linalg.norm(avg_enc)
                if norm > 0:
                    avg_enc = avg_enc / norm

            x1, y1, x2, y2 = best_face["bbox"]
            os.makedirs(CAPTURED_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_img = best_frame[y1:y2, x1:x2]
            path = os.path.join(CAPTURED_DIR, "{}_{}.jpg".format(name, ts))
            cv2.imwrite(path, face_img)

            if name not in self.known_names:
                self.known_names.append(name)
                self.known_encodings.append(avg_enc)
            else:
                idx = self.known_names.index(name)
                if avg_enc is not None:
                    self.known_encodings[idx] = avg_enc

            profile = {
                "name": get_val("name"),
                "age": get_val("age"),
                "job": get_val("job"),
                "desc": get_val("desc"),
                "photo": path,
                "added": ts,
            }
            self.profiles[name] = profile
            self._save_data()
            messagebox.showinfo("Saved", "Profile saved for '{}'!".format(name), parent=win)
            win.destroy()

        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=(20, 15))

        ctk.CTkButton(btn_frame, text="CAPTURE FACE", height=42,
                      fg_color="#238636", hover_color="#2ea043",
                      font=ctk.CTkFont(size=13, weight="bold"),
                      corner_radius=8, command=on_submit).pack(side="left", padx=(0, 8), expand=True, fill="x")

        ctk.CTkButton(btn_frame, text="CANCEL", height=42,
                      fg_color="#da3633", hover_color="#f85149",
                      font=ctk.CTkFont(size=13), corner_radius=8,
                      command=win.destroy).pack(side="right", padx=(8, 0), expand=True, fill="x")

    def _remove_face(self):
        if not self.known_names:
            messagebox.showinfo("Info", "No people saved!")
            return
        win = ctk.CTkToplevel(self)
        win.title("Remove Person")
        win.geometry("380x420")
        win.configure(fg_color="#0d1117")

        ctk.CTkLabel(win, text="SELECT PERSON TO REMOVE",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#f85149").pack(pady=(15, 10))

        lb = ctk.CTkTextbox(win, font=ctk.CTkFont(size=12), fg_color="#161b22",
                            text_color="#e6edf3", corner_radius=8)
        lb.pack(fill="both", expand=True, padx=15, pady=5)
        for n in self.known_names:
            lb.insert("end", n + "\n")

        def do_delete():
            try:
                sel = lb.get("anchor", "end").strip()
                if sel and sel in self.known_names:
                    idx = self.known_names.index(sel)
                    self.known_names.pop(idx)
                    self.known_encodings.pop(idx)
                    if sel in self.profiles:
                        del self.profiles[sel]
                    self._save_data()
                    lb.delete("1.0", "end")
                    for n in self.known_names:
                        lb.insert("end", n + "\n")
                    messagebox.showinfo("Removed", "'{}' removed.".format(sel))
            except:
                pass

        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkButton(btn_frame, text="DELETE", height=36,
                      fg_color="#da3633", hover_color="#f85149",
                      font=ctk.CTkFont(size=12), corner_radius=8,
                      command=do_delete).pack(side="left", expand=True, fill="x", padx=(0, 5))
        ctk.CTkButton(btn_frame, text="CLOSE", height=36,
                      fg_color="#30363d", hover_color="#484f58",
                      font=ctk.CTkFont(size=12), corner_radius=8,
                      command=win.destroy).pack(side="right", expand=True, fill="x", padx=(5, 0))

    def _list_faces(self):
        if not self.known_names:
            messagebox.showinfo("People", "No people saved.")
            return
        win = ctk.CTkToplevel(self)
        win.title("People Database")
        win.geometry("420x450")
        win.configure(fg_color="#0d1117")

        ctk.CTkLabel(win, text="PEOPLE DATABASE",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#58a6ff").pack(pady=(15, 10))

        scroll = ctk.CTkScrollableFrame(win, fg_color="transparent",
                                        scrollbar_button_color="#30363d")
        scroll.pack(fill="both", expand=True, padx=15, pady=5)

        for n in self.known_names:
            p = self.profiles.get(n, {})
            card = ctk.CTkFrame(scroll, fg_color="#161b22", corner_radius=8,
                                border_width=1, border_color="#30363d")
            card.pack(fill="x", pady=3, padx=2)

            ctk.CTkLabel(card, text=n, font=ctk.CTkFont(size=13, weight="bold"),
                         text_color="#e6edf3", anchor="w").pack(fill="x", padx=12, pady=(8, 2))

            parts = []
            if p.get("age"):
                parts.append("Age: {}".format(p["age"]))
            if p.get("job"):
                parts.append(p["job"])
            if p.get("desc"):
                parts.append(p["desc"])
            if parts:
                ctk.CTkLabel(card, text=" | ".join(parts),
                             font=ctk.CTkFont(size=10),
                             text_color="#8b949e", anchor="w").pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkButton(win, text="CLOSE", height=36,
                      fg_color="#30363d", hover_color="#484f58",
                      font=ctk.CTkFont(size=12), corner_radius=8,
                      command=win.destroy).pack(pady=10)

    def _screenshot(self):
        if not self.running:
            return
        with self.lock:
            frame = self.latest_frame
        if frame is not None:
            os.makedirs(CAPTURED_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(CAPTURED_DIR, "shot_{}.jpg".format(ts)), frame)
            messagebox.showinfo("Saved", "Screenshot saved!")

    def _open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            return
        if self.detect_faces and self.face_det and self.face_det.initialized:
            try:
                faces = self.face_det.detect(img)
                for f in faces:
                    x1, y1, x2, y2 = f["bbox"]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (129, 185, 16), 2)
                    name = "Unknown"
                    if self.face_enc and self.face_enc.initialized:
                        enc = self.face_enc.encode(img, x1, y1, x2, y2)
                        if enc is not None:
                            name, _ = self._match_face(enc)
                    cv2.putText(img, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (129, 185, 16), 2)
            except:
                pass
        win = ctk.CTkToplevel(self)
        win.title("Image Viewer")
        win.configure(fg_color="#0d1117")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        s = min(900 / pil.width, 700 / pil.height, 1.0)
        pil = pil.resize((int(pil.width * s), int(pil.height * s)))
        ph = ImageTk.PhotoImage(pil)
        lbl = ctk.CTkLabel(win, image=ph, text="")
        lbl.image = ph
        lbl.pack(padx=10, pady=10)

    def _quit(self):
        self.running = False
        if self.webcam:
            self.webcam.release()
        self.destroy()


if __name__ == "__main__":
    App()
