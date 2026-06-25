import cv2
import time
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT


class WebcamManager:
    def __init__(self, camera_index=None, width=None, height=None):
        self.camera_index = camera_index or CAMERA_INDEX
        self.width = width or FRAME_WIDTH
        self.height = height or FRAME_HEIGHT
        self.cap = None
        self.is_opened = False
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0

    def open(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"[WebcamManager] Cannot open camera {self.camera_index}")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.is_opened = True
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[WebcamManager] Camera opened: {actual_w}x{actual_h}")
            return True
        except Exception as e:
            print(f"[WebcamManager] Error opening camera: {e}")
            return False

    def read_frame(self):
        if not self.is_opened or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if ret:
            self._update_fps()
        return ret, frame

    def _update_fps(self):
        self.fps_counter += 1
        elapsed = time.time() - self.fps_timer
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_timer = time.time()

    def get_fps(self):
        return self.current_fps

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("[WebcamManager] Camera released.")

    def __del__(self):
        self.release()
