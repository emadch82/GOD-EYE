import cv2
import numpy as np
import os, sys, pickle, time, threading, collections, winsound
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
        model_path = ONNX_MODEL
        if not os.path.exists(model_path):
            return
        try:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"[ObjDet] Loaded: {os.path.basename(model_path)} ({os.path.getsize(model_path)/1024/1024:.0f}MB)")
        except:
            pass

    def detect(self, frame, skip=5):
        self.frame_count += 1
        if self.frame_count % skip != 0:
            return self.last_results
        if self.net is None:
            return []
        h, w = frame.shape[:2]
        inp_size = 320
        inp = cv2.resize(frame, (inp_size, inp_size))
        blob = cv2.dnn.blobFromImage(inp, 1.0 / 255.0, (inp_size, inp_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        detections = []
        sx, sy = w / inp_size, h / inp_size
        out2 = out.squeeze(0).T
        for row in out2:
            scores = row[4:]
            cls = int(np.argmax(scores))
            conf = float(scores[cls])
            if conf < 0.25:
                continue
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
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
        self.alarm_enabled = False
        self.recording = False
        self.video_writer = None
        self.detection_log = []
        self.last_alarm_time = 0
        self.detect_qr = False
        self.show_histogram = False
        self.blur_unknown = False
        self.night_mode = False
        self.face_track_ids = {}
        self._next_track_id = 1
        self.prev_frame_gray = None
        self.motion_detected = False
        self.zoom_level = 1.0
        self.zoom_center = None
        self._is_fullscreen = False
        self._sidebar_ref = None
        self.people_count = 0
        self.people_max = 0
        self.people_total = 0
        self._counted_ids = set()
        self.auto_snapshot = False
        self._last_snapshot_time = 0
        self.snapshot_count = 0
        self._det_thread_running = False
        self._det_result_faces = []
        self._det_result_objects = []
        self._det_frame = None
        self._det_lock = threading.Lock()
        self.smart_zoom = False
        self.zone_alert = False
        self.zone_points = []
        self.zone_active = False
        self.speed_tracker = False
        self._prev_positions = {}
        self.speed_data = {}
        self.line_crossing_enabled = False
        self.line_y = None
        self.line_direction = "horizontal"
        self.line_enter_count = 0
        self.line_exit_count = 0
        self._prev_faces_y = {}
        self.heatmap_enabled = False
        self.heatmap_data = None
        self.heatmap_alpha = 0.7
        self.multi_cam_enabled = False
        self.multi_cams = []
        self.multi_frames = [None, None, None, None]

        self._load_data()
        self._build()
        self.after(100, self._load_bg)
        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.bind("<Escape>", lambda e: self._toggle_fullscreen() if self._is_fullscreen else None)
        self.state("zoomed")
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

        sidebar_outer = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color="#161b22")
        sidebar_outer.grid(row=0, column=0, sticky="nsew")
        sidebar_outer.grid_propagate(False)

        sidebar = ctk.CTkScrollableFrame(sidebar_outer, fg_color="#161b22",
                                          scrollbar_button_color="#30363d",
                                          scrollbar_button_hover_color="#484f58")
        sidebar.pack(fill="both", expand=True)

        ctk.CTkLabel(sidebar, text="AI VISION PRO", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color="#58a6ff").pack(padx=10, pady=(10, 2))

        engine = "RUST" if RUST else "PYTHON"
        self.status_badge = ctk.CTkLabel(sidebar, text=f"Ready [{engine}]",
                                         font=ctk.CTkFont(size=10),
                                         text_color="#8b949e", fg_color="#21262d",
                                         corner_radius=4, padx=6, pady=2)
        self.status_badge.pack(padx=10, anchor="w", pady=(0, 8))

        self.btn_start = ctk.CTkButton(sidebar, text="Start Camera", height=32,
                                       fg_color="#238636", hover_color="#2ea043",
                                       font=ctk.CTkFont(size=11, weight="bold"),
                                       corner_radius=6, command=self._start)
        self.btn_start.pack(padx=10, pady=2, fill="x")

        btn_row = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_row.pack(padx=10, pady=2, fill="x")

        self.btn_stop = ctk.CTkButton(btn_row, text="Stop", height=28,
                                      fg_color="#da3633", hover_color="#f85149",
                                      font=ctk.CTkFont(size=10), corner_radius=6,
                                      command=self._stop)
        self.btn_stop.pack(side="left", padx=(0, 2), expand=True, fill="x")

        self.btn_pause = ctk.CTkButton(btn_row, text="Pause", height=28,
                                       fg_color="#9e6a03", hover_color="#d29922",
                                       font=ctk.CTkFont(size=10), corner_radius=6,
                                       command=self._pause)
        self.btn_pause.pack(side="right", padx=(2, 0), expand=True, fill="x")

        ctk.CTkFrame(sidebar, height=1, fg_color="#30363d").pack(fill="x", padx=10, pady=6)

        self.face_var = ctk.BooleanVar(value=True)
        self.obj_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(sidebar, text="Face Detection", variable=self.face_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#58a6ff").pack(padx=10, anchor="w", pady=1)
        ctk.CTkSwitch(sidebar, text="Object Detection", variable=self.obj_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#58a6ff").pack(padx=10, anchor="w", pady=1)
        self.enhanced_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Enhanced Detect", variable=self.enhanced_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#f0883e",
                      command=self._toggle_enhanced).pack(padx=10, anchor="w", pady=1)

        ctk.CTkFrame(sidebar, height=1, fg_color="#30363d").pack(fill="x", padx=10, pady=6)

        for text, cmd, color in [
            ("Add Person", self._add_face, "#1f6feb"),
            ("Remove Person", self._remove_face, "#8957e5"),
            ("List People", self._list_faces, "#30363d"),
            ("Screenshot", self._screenshot, "#30363d"),
            ("Open Image", self._open_image, "#30363d"),
            ("Compare Faces", self._compare_faces, "#30363d"),
            ("Show Logs", self._show_logs, "#30363d"),
            ("Gallery", self._show_gallery, "#30363d"),
            ("Export CSV", self._export_csv, "#30363d"),
        ]:
            ctk.CTkButton(sidebar, text=text, height=28,
                          fg_color=color, hover_color="#484f58",
                          text_color="#c9d1d9",
                          font=ctk.CTkFont(size=10), corner_radius=6,
                          command=cmd).pack(padx=10, pady=1, fill="x")

        ctk.CTkFrame(sidebar, height=1, fg_color="#30363d").pack(fill="x", padx=10, pady=6)

        self.alarm_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Unknown Alert", variable=self.alarm_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#f85149",
                      command=self._toggle_alarm).pack(padx=10, anchor="w", pady=1)
        self.rec_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Record Video", variable=self.rec_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#da3633",
                      command=self._toggle_record).pack(padx=10, anchor="w", pady=1)
        self.snap_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Auto Snapshot", variable=self.snap_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#3fb950",
                      command=self._toggle_snapshot).pack(padx=10, anchor="w", pady=1)
        self.blur_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Blur Unknown", variable=self.blur_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#a371f7",
                      command=self._toggle_blur).pack(padx=10, anchor="w", pady=1)
        self.night_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Night Mode", variable=self.night_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#d29922",
                      command=self._toggle_night).pack(padx=10, anchor="w", pady=1)
        self.hist_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Histogram", variable=self.hist_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#58a6ff",
                      command=self._toggle_histogram).pack(padx=10, anchor="w", pady=1)

        ctk.CTkFrame(sidebar, height=1, fg_color="#30363d").pack(fill="x", padx=10, pady=6)

        self.stat_people_live = ctk.CTkLabel(sidebar, text="People: 0",
                                             font=ctk.CTkFont(size=11, weight="bold"),
                                             text_color="#f0883e", fg_color="#21262d",
                                             corner_radius=4, padx=6, pady=2)
        self.stat_people_live.pack(padx=10, anchor="w", pady=1)
        self.stat_people_max = ctk.CTkLabel(sidebar, text="Max: 0",
                                             font=ctk.CTkFont(size=10),
                                             text_color="#8b949e")
        self.stat_people_max.pack(padx=10, anchor="w", pady=0)
        self.stat_people_total = ctk.CTkLabel(sidebar, text="Total: 0",
                                               font=ctk.CTkFont(size=10),
                                               text_color="#8b949e")
        self.stat_people_total.pack(padx=10, anchor="w", pady=(0, 4))

        self.line_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Line Crossing", variable=self.line_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#f0883e",
                      command=self._toggle_line).pack(padx=10, anchor="w", pady=1)
        self.line_enter_label = ctk.CTkLabel(sidebar, text="  Enter: 0",
                                              font=ctk.CTkFont(size=10),
                                              text_color="#3fb950")
        self.line_enter_label.pack(padx=10, anchor="w")
        self.line_exit_label = ctk.CTkLabel(sidebar, text="  Exit: 0",
                                             font=ctk.CTkFont(size=10),
                                             text_color="#f85149")
        self.line_exit_label.pack(padx=10, anchor="w", pady=(0, 2))

        self.heatmap_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Heatmap", variable=self.heatmap_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#da3633",
                      command=self._toggle_heatmap).pack(padx=10, anchor="w", pady=1)

        self.multi_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Multi Camera", variable=self.multi_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#1f6feb",
                      command=self._toggle_multi).pack(padx=10, anchor="w", pady=1)

        ctk.CTkFrame(sidebar, height=1, fg_color="#30363d").pack(fill="x", padx=10, pady=4)

        self.smart_zoom_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Smart Zoom", variable=self.smart_zoom_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#3fb950",
                      command=self._toggle_smart_zoom).pack(padx=10, anchor="w", pady=1)

        self.zone_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Zone Alert", variable=self.zone_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#f85149",
                      command=self._toggle_zone).pack(padx=10, anchor="w", pady=1)

        self.speed_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(sidebar, text="Speed Tracker", variable=self.speed_var,
                      font=ctk.CTkFont(size=11), text_color="#c9d1d9",
                      fg_color="#30363d", progress_color="#d29922",
                      command=self._toggle_speed).pack(padx=10, anchor="w", pady=1)

        ctk.CTkButton(sidebar, text="Fullscreen (F11)", height=28,
                      fg_color="#30363d", hover_color="#484f58",
                      text_color="#c9d1d9",
                      font=ctk.CTkFont(size=10), corner_radius=6,
                      command=self._toggle_fullscreen).pack(padx=10, pady=1, fill="x")

        ctk.CTkButton(sidebar, text="Quit", height=28,
                      fg_color="#da3633", hover_color="#f85149",
                      font=ctk.CTkFont(size=10), corner_radius=6,
                      command=self._quit).pack(padx=10, pady=(6, 10), fill="x")

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
        self.latest_frame = None
        self.webcam = cv2.VideoCapture(CAMERA_INDEX)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.webcam.isOpened():
            messagebox.showerror("Error", "Camera not available!")
            self.webcam = None
            return
        self.running = True
        self.paused = False
        self.fps_q.clear()
        self.face_track_ids = {}
        self._counted_ids = set()
        self._prev_faces_y = {}
        self._enc_cache = {}
        self._enc_counter = 0
        self.prev_frame_gray = None
        self._det_result_faces = []
        self._det_result_objects = []
        if self.heatmap_data is not None:
            self.heatmap_data *= 0
        self.status_badge.configure(text="REC", fg_color="#da3633", text_color="#ffffff")
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._detect_worker, daemon=True).start()
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

    def _detect_worker(self):
        while self.running:
            try:
                with self.lock:
                    frame = self.latest_frame
                if frame is None:
                    time.sleep(0.01)
                    continue
                faces = []
                objects = []
                if self.face_var.get() and self.face_det and self.face_det.initialized:
                    try:
                        faces = self.face_det.detect(frame)
                    except:
                        pass
                if self.obj_var.get() and self.obj_det:
                    try:
                        objects = self.obj_det.detect(frame, skip=1)
                    except:
                        pass
                with self._det_lock:
                    self._det_result_faces = faces
                    self._det_result_objects = objects
            except:
                pass
            time.sleep(0.01)

    def _stop(self):
        self.running = False
        time.sleep(0.05)
        if self.webcam:
            try:
                self.webcam.release()
            except:
                pass
            self.webcam = None
        if self.recording and self.video_writer:
            try:
                self.video_writer.release()
            except:
                pass
            self.video_writer = None
            self.recording = False
            self.rec_var.set(False)
        self.latest_frame = None
        self.prev_frame_gray = None
        self.motion_detected = False
        self.people_count = 0
        self.face_track_ids = {}
        self._counted_ids = set()
        self._prev_faces_y = {}
        self._enc_cache = {}
        self.status_badge.configure(text="STOPPED", fg_color="#da3633", text_color="#ffffff")
        self.canvas_label.configure(image=None, text="Camera Off")
        self.stat_fps.configure(text="FPS: --")
        self.stat_faces.configure(text="Faces: 0")
        self.stat_objects.configure(text="Objects: 0")
        self.stat_people_live.configure(text="People: 0")
        self.stat_people_max.configure(text="Max: 0")
        self.stat_people_total.configure(text="Total: 0")
        self.stat_people_total.configure(text="Total: 0")

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

        if self.night_mode:
            frame = self._apply_night_mode(frame)

        self.motion_detected = self._detect_motion(frame)

        with self._det_lock:
            faces = list(self._det_result_faces)
            objects = list(self._det_result_objects)

        face_infos = []
        unknown_bboxes = []
        self._enc_counter += 1

        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            color = (129, 185, 16)

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

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            track_id = None
            best_dist = 80
            for tid, pos in self.face_track_ids.items():
                dx = cx - pos[0]
                dy = cy - pos[1]
                d = (dx * dx + dy * dy) ** 0.5
                if d < best_dist:
                    best_dist = d
                    track_id = tid
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
            self.face_track_ids[track_id] = (cx, cy)

            face_infos.append({"name": name, "lines": profile_lines, "known": is_known, "track_id": track_id})
            if not is_known:
                unknown_bboxes.append((x1, y1, x2, y2))

            if is_known:
                color = (129, 185, 16)
            elif self.blur_unknown and self.blur_var.get():
                self._blur_face_region(frame, x1, y1, x2, y2)
                color = (163, 113, 247)
            else:
                color = (68, 68, 239)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cl = 12
            cv2.line(frame, (x1, y1), (x1 + cl, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + cl), color, 3)
            cv2.line(frame, (x2, y1), (x2 - cl, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + cl), color, 3)
            cv2.line(frame, (x1, y2), (x1 + cl, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - cl), color, 3)
            cv2.line(frame, (x2, y2), (x2 - cl, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - cl), color, 3)

            status = "KNOWN" if is_known else "UNKNOWN"
            sc = (129, 185, 16) if is_known else (68, 68, 239)
            cv2.putText(frame, status, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1, cv2.LINE_AA)

            if is_known:
                cv2.putText(frame, name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (129, 185, 16), 1, cv2.LINE_AA)

        for o in objects:
            x1, y1, x2, y2 = o["bbox"]
            color = (6, 182, 212) if o["name"] in ["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = "{} {:.0%}".format(o["name"], o["confidence"])
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if self.motion_detected:
            cv2.putText(frame, "MOTION DETECTED", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        current_ids = set()
        for fi in face_infos:
            if fi.get("track_id") is not None:
                current_ids.add(fi["track_id"])

        self.people_count = len(face_infos)
        if self.people_count > self.people_max:
            self.people_max = self.people_count
        new_ids = current_ids - self._counted_ids
        self.people_total += len(new_ids)
        self._counted_ids.update(current_ids)

        disappeared = self._counted_ids - current_ids
        if len(disappeared) > 5:
            self._counted_ids = current_ids

        face_centers = {}
        for fi in face_infos:
            if fi.get("track_id") is not None:
                face_centers[fi["track_id"]] = self.face_track_ids.get(fi["track_id"], (0, 0))

        self._track_line_crossing(face_centers)
        self._update_heatmap(face_centers)

        unknown_count = sum(1 for fi in face_infos if not fi["known"])
        if unknown_count > 0 and self.alarm_enabled:
            now = time.time()
            if now - self.last_alarm_time > 10:
                self.last_alarm_time = now
                threading.Thread(target=self._play_alarm, daemon=True).start()

        if unknown_count > 0 and self.auto_snapshot:
            for bbox in unknown_bboxes:
                self._auto_capture_unknown(frame, *bbox)

        if face_infos and self._enc_counter % 30 == 0:
            for fi in face_infos:
                self._log_detection(fi["name"], fi["known"])

        if self.recording and self.video_writer:
            display_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            self.video_writer.write(display_frame)

        dt = time.time() - t0
        if dt > 0:
            self.fps_q.append(1.0 / dt)
        fps = sum(self.fps_q) / len(self.fps_q) if self.fps_q else 0

        if self.smart_zoom:
            frame = self._apply_smart_zoom(frame, faces)

        self._draw_zone(frame)
        zone_alert = self._check_zone_alert(face_centers)
        if zone_alert:
            cv2.putText(frame, "ALERT: ZONE BREACH!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.alarm_enabled:
                now = time.time()
                if now - self.last_alarm_time > 5:
                    self.last_alarm_time = now
                    threading.Thread(target=self._play_alarm, daemon=True).start()

        speeds = self._update_speed(face_centers)
        self._draw_speed(frame, face_centers, speeds)

        scene_desc = self._generate_scene_description(face_infos, objects)

        if self.show_histogram:
            self._draw_histogram(frame)

        self._draw_line_crossing(frame)

        if self.heatmap_enabled:
            self._draw_heatmap(frame)

        if self.multi_cam_enabled and self.multi_cams:
            grid = self._draw_multi_grid(frame)
            rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        else:
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
        self.stat_people_live.configure(text="People: {}".format(self.people_count))
        self.stat_people_max.configure(text="Max: {}".format(self.people_max))
        self.stat_people_total.configure(text="Total: {}".format(self.people_total))
        self.s_time.configure(text=datetime.now().strftime("%H:%M:%S"))
        parts = ["{}x{}".format(self.cam_w, self.cam_h)]
        parts.append("RUST" if RUST else "PYTHON")
        if self.motion_detected:
            parts.append("MOTION")
        if self.night_mode:
            parts.append("NIGHT")
        self.s_info.configure(text=" | ".join(parts))

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

    def _toggle_alarm(self):
        self.alarm_enabled = self.alarm_var.get()
        if self.alarm_enabled:
            self.s_info.configure(text="ALARM: ON - Unknown faces will trigger alert")
        else:
            self.s_info.configure(text="ALARM: OFF")

    def _toggle_histogram(self):
        self.show_histogram = self.hist_var.get()
        self.s_info.configure(text="Histogram: " + ("ON" if self.show_histogram else "OFF"))

    def _toggle_enhanced(self):
        enabled = self.enhanced_var.get()
        if self.face_det:
            self.face_det._multi_scale = enabled
            self.face_det._flip_augment = enabled
        self.s_info.configure(text="Enhanced Detection: " + ("ON" if enabled else "OFF"))

    def _toggle_blur(self):
        self.blur_unknown = self.blur_var.get()
        self.s_info.configure(text="Blur Unknown: " + ("ON" if self.blur_unknown else "OFF"))

    def _toggle_snapshot(self):
        self.auto_snapshot = self.snap_var.get()
        if self.auto_snapshot:
            os.makedirs(CAPTURED_DIR, exist_ok=True)
        self.s_info.configure(text="Auto Snapshot: " + ("ON" if self.auto_snapshot else "OFF"))

    def _auto_capture_unknown(self, frame, x1, y1, x2, y2):
        now = time.time()
        if now - self._last_snapshot_time < 5:
            return
        self._last_snapshot_time = now
        self.snapshot_count += 1

        pad = 30
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(frame.shape[1], x2 + pad)
        cy2 = min(frame.shape[0], y2 + pad)
        face_crop = frame[cy1:cy2, cx1:cx2]

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CAPTURED_DIR, "unknown_{}_{}.jpg".format(ts, self.snapshot_count))
        cv2.imwrite(path, face_crop)

        self._log_detection("Unknown (auto)", False)

    def _toggle_night(self):
        self.night_mode = self.night_var.get()
        self.s_info.configure(text="Night Mode: " + ("ON" if self.night_mode else "OFF"))

    def _toggle_fullscreen(self):
        self._is_fullscreen = not self._is_fullscreen
        if self._is_fullscreen:
            self.state("zoomed")
        else:
            self.state("normal")
            self.geometry("1920x1080")

    def _toggle_line(self):
        self.line_crossing_enabled = self.line_var.get()
        if self.line_crossing_enabled:
            self.line_y = None
            self.line_enter_count = 0
            self.line_exit_count = 0
            self._prev_faces_y = {}
            self.s_info.configure(text="LINE: Click on camera to set crossing line")
            self.canvas_label.bind("<Button-1>", self._set_line_click)
        else:
            self.canvas_label.unbind("<Button-1>")
            self.s_info.configure(text="Line Crossing: OFF")

    def _set_line_click(self, event):
        if not self.line_crossing_enabled:
            return
        cw = self.canvas_label.winfo_width()
        ch = self.canvas_label.winfo_height()
        img_h = ch
        rel_y = event.y / img_h
        self.line_y = rel_y
        self.s_info.configure(text="LINE SET at {:.0%} height".format(rel_y))

    def _draw_line_crossing(self, frame):
        if not self.line_crossing_enabled or self.line_y is None:
            return
        h, w = frame.shape[:2]
        ly = int(self.line_y * h)
        cv2.line(frame, (0, ly), (w, ly), (0, 165, 255), 2)
        cv2.putText(frame, "CROSSING LINE", (10, ly - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    def _track_line_crossing(self, face_centers):
        if not self.line_crossing_enabled or self.line_y is None:
            return
        h = self.cam_h
        line_pixel = self.line_y * h

        for tid, (cx, cy) in face_centers.items():
            if tid in self._prev_faces_y:
                prev_y = self._prev_faces_y[tid]
                if prev_y < line_pixel <= cy:
                    self.line_enter_count += 1
                elif prev_y > line_pixel >= cy:
                    self.line_exit_count += 1
            self._prev_faces_y[tid] = cy

        active_ids = set(face_centers.keys())
        self._prev_faces_y = {k: v for k, v in self._prev_faces_y.items() if k in active_ids}

        self.line_enter_label.configure(text="  Enter: {}".format(self.line_enter_count))
        self.line_exit_label.configure(text="  Exit: {}".format(self.line_exit_count))

    def _toggle_heatmap(self):
        self.heatmap_enabled = self.heatmap_var.get()
        if self.heatmap_enabled:
            self.heatmap_data = np.zeros((self.cam_h, self.cam_w), dtype=np.float32)
        self.s_info.configure(text="Heatmap: " + ("ON" if self.heatmap_enabled else "OFF"))

    def _update_heatmap(self, face_centers):
        if not self.heatmap_enabled or self.heatmap_data is None:
            return
        for tid, (cx, cy) in face_centers.items():
            if 0 <= cy < self.heatmap_data.shape[0] and 0 <= cx < self.heatmap_data.shape[1]:
                y1 = max(0, cy - 30)
                y2 = min(self.heatmap_data.shape[0], cy + 30)
                x1 = max(0, cx - 30)
                x2 = min(self.heatmap_data.shape[1], cx + 30)
                self.heatmap_data[y1:y2, x1:x2] += 0.3

    def _draw_heatmap(self, frame):
        if not self.heatmap_enabled or self.heatmap_data is None:
            return
        h, w = frame.shape[:2]
        hm = cv2.resize(self.heatmap_data, (w, h))
        hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.6, hm_color, 0.4, 0)
        frame[:] = blended
        cv2.putText(frame, "HEATMAP", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.heatmap_data *= 0.995

    def _toggle_multi(self):
        self.multi_cam_enabled = self.multi_var.get()
        if self.multi_cam_enabled:
            self.multi_cams = []
            for idx in range(4):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    self.multi_cams.append(cap)
                else:
                    cap.release()
            if not self.multi_cams:
                self.multi_var.set(False)
                self.multi_cam_enabled = False
                messagebox.showinfo("Info", "No additional cameras found!")
                return
            self.s_info.configure(text="Multi Camera: {} cams".format(len(self.multi_cams)))
        else:
            for cap in self.multi_cams:
                try:
                    cap.release()
                except:
                    pass
            self.multi_cams = []
            self.s_info.configure(text="Multi Camera: OFF")

    def _draw_multi_grid(self, frame):
        if not self.multi_cam_enabled or not self.multi_cams:
            return frame
        grid_size = 2
        cells = []
        for cap in self.multi_cams:
            ret, f = cap.read()
            if ret:
                f = cv2.flip(f, 1)
                f = cv2.resize(f, (FRAME_WIDTH // grid_size, FRAME_HEIGHT // grid_size))
                cells.append(f)
            else:
                cells.append(np.zeros((FRAME_HEIGHT // grid_size, FRAME_WIDTH // grid_size, 3), dtype=np.uint8))

        while len(cells) < 4:
            cells.append(np.zeros((FRAME_HEIGHT // grid_size, FRAME_WIDTH // grid_size, 3), dtype=np.uint8))

        top = np.hstack(cells[:2])
        bottom = np.hstack(cells[2:4])
        grid = np.vstack([top, bottom])
        return grid

    def _toggle_smart_zoom(self):
        self.smart_zoom = self.smart_zoom_var.get()
        self.s_info.configure(text="Smart Zoom: " + ("ON" if self.smart_zoom else "OFF"))

    def _apply_smart_zoom(self, frame, faces):
        if not self.smart_zoom or not faces:
            return frame
        best = max(faces, key=lambda f: f.get("confidence", 0))
        x1, y1, x2, y2 = best["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        fw, fh = frame.shape[1], frame.shape[0]
        zoom = 2.0
        zw = int(fw / zoom)
        zh = int(fh / zoom)
        zx1 = max(0, cx - zw // 2)
        zy1 = max(0, cy - zh // 2)
        zx2 = min(fw, zx1 + zw)
        zy2 = min(fh, zy1 + zh)
        cropped = frame[zy1:zy2, zx1:zx2]
        if cropped.size == 0:
            return frame
        return cv2.resize(cropped, (fw, fh), interpolation=cv2.INTER_LINEAR)

    def _toggle_zone(self):
        self.zone_alert = self.zone_var.get()
        if self.zone_alert:
            self.zone_points = []
            self.zone_active = True
            self.s_info.configure(text="ZONE: Click 3+ points to draw zone, right-click to finish")
            self.canvas_label.bind("<Button-1>", self._zone_click)
            self.canvas_label.bind("<Button-3>", self._zone_finish)
        else:
            self.zone_points = []
            self.zone_active = False
            self.canvas_label.unbind("<Button-1>")
            self.canvas_label.unbind("<Button-3>")
            self.s_info.configure(text="Zone Alert: OFF")

    def _zone_click(self, event):
        if not self.zone_active:
            return
        cw = self.canvas_label.winfo_width()
        ch = self.canvas_label.winfo_height()
        rx = event.x / cw
        ry = event.y / ch
        self.zone_points.append((rx, ry))
        self.s_info.configure(text="ZONE: {} points (right-click to finish)".format(len(self.zone_points)))

    def _zone_finish(self, event):
        if len(self.zone_points) >= 3:
            self.zone_active = False
            self.s_info.configure(text="ZONE SET: {} points".format(len(self.zone_points)))

    def _draw_zone(self, frame):
        if not self.zone_alert or len(self.zone_points) < 3:
            return
        h, w = frame.shape[:2]
        pts = np.array([(int(px * w), int(py * h)) for px, py in self.zone_points], np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 100))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        cv2.putText(frame, "RESTRICTED ZONE", (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def _check_zone_alert(self, face_centers):
        if not self.zone_alert or len(self.zone_points) < 3:
            return False
        h, w = self.cam_h, self.cam_w
        zone = np.array([(int(px * w), int(py * h)) for px, py in self.zone_points], np.int32)
        for tid, (cx, cy) in face_centers.items():
            result = cv2.pointPolygonTest(zone, (float(cx), float(cy)), False)
            if result >= 0:
                return True
        return False

    def _toggle_speed(self):
        self.speed_tracker = self.speed_var.get()
        if self.speed_tracker:
            self._prev_positions = {}
            self.speed_data = {}
        self.s_info.configure(text="Speed Tracker: " + ("ON" if self.speed_tracker else "OFF"))

    def _update_speed(self, face_centers):
        if not self.speed_tracker:
            return {}
        speeds = {}
        for tid, (cx, cy) in face_centers.items():
            if tid in self._prev_positions:
                px, py = self._prev_positions[tid]
                dx = cx - px
                dy = cy - py
                dist = (dx * dx + dy * dy) ** 0.5
                speed_px = dist * 30
                speeds[tid] = speed_px
            self._prev_positions[tid] = (cx, cy)
        return speeds

    def _draw_speed(self, frame, face_centers, speeds):
        if not self.speed_tracker:
            return
        h, w = frame.shape[:2]
        for tid, (cx, cy) in face_centers.items():
            if tid in speeds:
                spd = speeds[tid]
                color = (0, 255, 0) if spd < 100 else (0, 165, 255) if spd < 300 else (0, 0, 255)
                cv2.putText(frame, "{:.0f}px/s".format(spd), (cx - 30, cy - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _generate_scene_description(self, face_infos, objects):
        parts = []
        if face_infos:
            known = sum(1 for f in face_infos if f["known"])
            unknown = len(face_infos) - known
            if known > 0:
                names = [f["name"] for f in face_infos if f["known"]]
                parts.append("{} known ({})".format(known, ", ".join(names[:3])))
            if unknown > 0:
                parts.append("{} unknown".format(unknown))
        obj_names = {}
        for o in objects:
            n = o["name"]
            obj_names[n] = obj_names.get(n, 0) + 1
        for name, count in obj_names.items():
            parts.append("{}x {}".format(count, name))
        if not parts:
            return "Empty scene"
        return ", ".join(parts)

    def _draw_histogram(self, frame):
        h, w = frame.shape[:2]
        hist_h, hist_w = 100, 200
        x_off = w - hist_w - 15
        y_off = 15

        overlay = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            hist = cv2.calcHist([frame], [i], None, [64], [0, 256])
            cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
            pts = np.array([[int(x * hist_w / 64), hist_h - int(hist[x][0])]
                            for x in range(64)], np.int32)
            cv2.polylines(overlay, [pts], False, color, 1)

        roi = frame[y_off:y_off + hist_h, x_off:x_off + hist_w]
        if roi.shape[0] == hist_h and roi.shape[1] == hist_w:
            blended = cv2.addWeighted(roi, 0.3, overlay, 0.7, 0)
            frame[y_off:y_off + hist_h, x_off:x_off + hist_w] = blended
        cv2.putText(frame, "RGB Histogram", (x_off, y_off + hist_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    def _apply_night_mode(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.prev_frame_gray is None:
            self.prev_frame_gray = gray
            return False
        delta = cv2.absdiff(self.prev_frame_gray, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion = cv2.countNonZero(thresh) > 5000
        self.prev_frame_gray = gray
        return motion

    def _blur_face_region(self, frame, x1, y1, x2, y2):
        pad = 10
        fx1 = max(0, x1 - pad)
        fy1 = max(0, y1 - pad)
        fx2 = min(frame.shape[1], x2 + pad)
        fy2 = min(frame.shape[0], y2 + pad)
        roi = frame[fy1:fy2, fx1:fx2]
        blurred = cv2.GaussianBlur(roi, (51, 51), 30)
        frame[fy1:fy2, fx1:fx2] = blurred

    def _show_gallery(self):
        win = ctk.CTkToplevel(self)
        win.title("Screenshot Gallery")
        win.geometry("800x600")
        win.configure(fg_color="#0d1117")

        ctk.CTkLabel(win, text="SCREENSHOT GALLERY",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#58a6ff").pack(pady=(15, 10))

        scroll = ctk.CTkScrollableFrame(win, fg_color="transparent",
                                        scrollbar_button_color="#30363d")
        scroll.pack(fill="both", expand=True, padx=15, pady=5)

        if not os.path.exists(CAPTURED_DIR):
            ctk.CTkLabel(scroll, text="No screenshots yet",
                         font=ctk.CTkFont(size=11), text_color="#484f58").pack(pady=20)
            ctk.CTkButton(win, text="CLOSE", height=36, fg_color="#30363d",
                          hover_color="#484f58", font=ctk.CTkFont(size=12),
                          corner_radius=8, command=win.destroy).pack(pady=10)
            return

        files = sorted([f for f in os.listdir(CAPTURED_DIR)
                        if f.startswith("shot_") and f.endswith((".jpg", ".png"))],
                       reverse=True)

        if not files:
            ctk.CTkLabel(scroll, text="No screenshots found",
                         font=ctk.CTkFont(size=11), text_color="#484f58").pack(pady=20)
            ctk.CTkButton(win, text="CLOSE", height=36, fg_color="#30363d",
                          hover_color="#484f58", font=ctk.CTkFont(size=12),
                          corner_radius=8, command=win.destroy).pack(pady=10)
            return

        grid = ctk.CTkFrame(scroll, fg_color="transparent")
        grid.pack(fill="x")

        photos = []
        for i, fname in enumerate(files[:20]):
            path = os.path.join(CAPTURED_DIR, fname)
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                pil = pil.resize((150, 120), Image.LANCZOS)
                photo = ImageTk.PhotoImage(pil)
                photos.append(photo)

                row, col = divmod(i, 4)
                cell = ctk.CTkFrame(grid, fg_color="#161b22", corner_radius=8)
                cell.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

                lbl = ctk.CTkLabel(cell, image=photo, text="")
                lbl.pack(padx=5, pady=5)
                lbl.image = photo

                ts = fname.replace("shot_", "").replace(".jpg", "").replace(".png", "")
                ctk.CTkLabel(cell, text=ts, font=ctk.CTkFont(size=9),
                             text_color="#8b949e").pack(pady=(0, 5))

            except:
                pass

        for c in range(4):
            grid.columnconfigure(c, weight=1)

        ctk.CTkButton(win, text="CLOSE", height=36, fg_color="#30363d",
                      hover_color="#484f58", font=ctk.CTkFont(size=12),
                      corner_radius=8, command=win.destroy).pack(pady=10)

    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="detections_{}.csv".format(datetime.now().strftime("%Y%m%d"))
        )
        if not path:
            return

        all_logs = list(self.detection_log)
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "detections.log")
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and line.startswith("["):
                        ts_end = line.find("]")
                        if ts_end > 0:
                            ts = line[1:ts_end]
                            rest = line[ts_end + 2:]
                            paren = rest.rfind(" (")
                            if paren > 0:
                                name = rest[:paren]
                                status = rest[paren + 2:-1]
                                all_logs.append({
                                    "time": ts,
                                    "name": name,
                                    "known": status == "KNOWN"
                                })

        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Name", "Status"])
            for entry in all_logs:
                writer.writerow([
                    entry["time"],
                    entry["name"],
                    "KNOWN" if entry["known"] else "UNKNOWN"
                ])

        messagebox.showinfo("Exported", "Logs exported to:\n{}".format(path))

    def _play_alarm(self):
        try:
            winsound.Beep(1000, 300)
            time.sleep(0.1)
            winsound.Beep(1500, 300)
            time.sleep(0.1)
            winsound.Beep(2000, 500)
        except:
            pass

    def _toggle_record(self):
        if self.rec_var.get():
            if not self.running:
                self.rec_var.set(False)
                messagebox.showinfo("Info", "Start camera first!")
                return
            os.makedirs(CAPTURED_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(CAPTURED_DIR, "rec_{}.avi".format(ts))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(path, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))
            self.recording = True
            self.s_info.configure(text="RECORDING: {}".format(path))
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.s_info.configure(text="Recording stopped")

    def _log_detection(self, name, is_known):
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "known": is_known,
        }
        self.detection_log.append(entry)
        if len(self.detection_log) > 500:
            self.detection_log = self.detection_log[-500:]
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "detections.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("[{}] {} ({})\n".format(
                entry["time"], entry["name"],
                "KNOWN" if is_known else "UNKNOWN"))

    def _show_logs(self):
        win = ctk.CTkToplevel(self)
        win.title("Detection Logs")
        win.geometry("500x450")
        win.configure(fg_color="#0d1117")

        ctk.CTkLabel(win, text="DETECTION HISTORY",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#58a6ff").pack(pady=(15, 10))

        scroll = ctk.CTkScrollableFrame(win, fg_color="transparent",
                                        scrollbar_button_color="#30363d")
        scroll.pack(fill="both", expand=True, padx=15, pady=5)

        logs = self.detection_log[-100:][::-1]
        if not logs:
            log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "detections.log")
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-100:][::-1]
                for line in lines:
                    line = line.strip()
                    if line:
                        is_known = "KNOWN" in line
                        color = "#3fb950" if is_known else "#f85149"
                        ctk.CTkLabel(scroll, text=line, font=ctk.CTkFont(size=11),
                                     text_color=color, anchor="w").pack(fill="x", pady=1)
                ctk.CTkButton(win, text="CLOSE", height=36, fg_color="#30363d",
                              hover_color="#484f58", font=ctk.CTkFont(size=12),
                              corner_radius=8, command=win.destroy).pack(pady=10)
                return

        for entry in logs:
            color = "#3fb950" if entry["known"] else "#f85149"
            status = "KNOWN" if entry["known"] else "UNKNOWN"
            text = "[{}] {} - {}".format(entry["time"], entry["name"], status)
            ctk.CTkLabel(scroll, text=text, font=ctk.CTkFont(size=11),
                         text_color=color, anchor="w").pack(fill="x", pady=1)

        ctk.CTkButton(win, text="CLOSE", height=36, fg_color="#30363d",
                      hover_color="#484f58", font=ctk.CTkFont(size=12),
                      corner_radius=8, command=win.destroy).pack(pady=10)

    def _compare_faces(self):
        win = ctk.CTkToplevel(self)
        win.title("Compare Faces")
        win.geometry("500x400")
        win.configure(fg_color="#0d1117")

        ctk.CTkLabel(win, text="COMPARE TWO FACES",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#58a6ff").pack(pady=(15, 10))

        result_label = ctk.CTkLabel(win, text="Select two images to compare",
                                    font=ctk.CTkFont(size=12),
                                    text_color="#8b949e")
        result_label.pack(pady=10)

        images = [None, None]
        previews = [None, None]

        frame_row = ctk.CTkFrame(win, fg_color="transparent")
        frame_row.pack(fill="x", padx=20, pady=10)

        slots = []
        for i in range(2):
            slot = ctk.CTkFrame(frame_row, fg_color="#161b22", corner_radius=8,
                                width=180, height=180)
            slot.pack(side="left", expand=True, fill="both", padx=5)
            slot.pack_propagate(False)
            lbl = ctk.CTkLabel(slot, text="Image {}".format(i+1),
                               font=ctk.CTkFont(size=11), text_color="#484f58")
            lbl.pack(expand=True)
            slots.append((slot, lbl))

        def load_image(idx):
            path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp")])
            if not path:
                return
            img = cv2.imread(path)
            if img is None:
                return
            images[idx] = img
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil = pil.resize((170, 170), Image.LANCZOS)
            photo = ImageTk.PhotoImage(pil)
            slots[idx][0].configure(fg_color="#0d1117")
            slots[idx][1].configure(image=photo, text="")
            slots[idx][1].image = photo

        def do_compare():
            if images[0] is None or images[1] is None:
                messagebox.showinfo("Info", "Load both images first!", parent=win)
                return
            if not self.face_enc or not self.face_enc.initialized:
                messagebox.showinfo("Info", "Face encoder not ready!", parent=win)
                return

            encs = []
            for img in images:
                faces = self.face_det.detect(img) if self.face_det else []
                if faces:
                    best = max(faces, key=lambda f: f["confidence"])
                    x1, y1, x2, y2 = best["bbox"]
                    lm = best.get("landmarks", [])
                    enc = self.face_enc.encode(img, x1, y1, x2, y2, lm)
                    encs.append(enc)
                else:
                    encs.append(None)

            if encs[0] is None or encs[1] is None:
                result_label.configure(text="No face found in one/both images!",
                                       text_color="#f85149")
                return

            score = self.face_enc.match(encs[0], encs[1])
            if score > 0.45:
                result_label.configure(
                    text="SAME PERSON ({:.1%} match)".format(score),
                    text_color="#3fb950")
            else:
                result_label.configure(
                    text="DIFFERENT PEOPLE ({:.1%} match)".format(score),
                    text_color="#f85149")

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkButton(btn_row, text="Load Image 1", height=36,
                      fg_color="#1f6feb", hover_color="#388bfd",
                      font=ctk.CTkFont(size=12), corner_radius=8,
                      command=lambda: load_image(0)).pack(side="left", expand=True, fill="x", padx=3)
        ctk.CTkButton(btn_row, text="Load Image 2", height=36,
                      fg_color="#1f6feb", hover_color="#388bfd",
                      font=ctk.CTkFont(size=12), corner_radius=8,
                      command=lambda: load_image(1)).pack(side="right", expand=True, fill="x", padx=3)

        ctk.CTkButton(win, text="COMPARE", height=42,
                      fg_color="#238636", hover_color="#2ea043",
                      font=ctk.CTkFont(size=13, weight="bold"), corner_radius=8,
                      command=do_compare).pack(pady=15, padx=60, fill="x")

        ctk.CTkButton(win, text="CLOSE", height=36, fg_color="#30363d",
                      hover_color="#484f58", font=ctk.CTkFont(size=12),
                      corner_radius=8, command=win.destroy).pack(pady=(0, 10))

    def _quit(self):
        self.running = False
        if self.webcam:
            self.webcam.release()
        if self.video_writer:
            self.video_writer.release()
        self.destroy()


if __name__ == "__main__":
    App()
