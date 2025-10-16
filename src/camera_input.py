# 카메라 프레임을 안정적으로 읽어 vision_engine에 전달하는 모듈
import cv2

class CameraInput:
    def __init__(self, index=0, width=None, height=None):
        self.index = index
        self.cap = None

        # Use camera's native resolution if width/height are not provided
        if width is None or height is None:
            temp = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if temp.isOpened():
                native_w = int(temp.get(cv2.CAP_PROP_FRAME_WIDTH))
                native_h = int(temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                temp.release()
            else:
                native_w, native_h = 1280, 720  # fallback
            self.width = native_w if width is None else width
            self.height = native_h if height is None else height
        else:
            self.width = width
            self.height = height

    def open(self):
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera.")

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_frame(self):
        if self.cap is None:
            raise RuntimeError("Camera is not opened. Call open() before read_frame().")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Error: Could not read frame.")
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None