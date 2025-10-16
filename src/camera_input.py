import cv2
import os
import re
import subprocess

# 카메라 프레임을 안정적으로 읽어 vision_engine에 전달하는 모듈

def _list_resolutions_v4l2(index: int):
    dev = f"/dev/video{index}"
    if not os.path.exists(dev):
        return []
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", f"--device={dev}", "--list-formats-ext"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return []
    sizes = set()
    for w, h in re.findall(r'(\d{3,5})x(\d{3,5})', out):
        sizes.add((int(w), int(h)))
    return sorted(sizes, key=lambda wh: (wh[0]*wh[1], wh[0]), reverse=True)

class CameraInput:
    def __init__(self, index=0, width=None, height=None):
        self.index = index
        self.cap = None
        # 0 또는 None이면 자동 최대 해상도 선택 로직 사용
        auto = (width in (None, 0)) or (height in (None, 0))
        if auto:
            self.width = 0
            self.height = 0
        else:
            self.width = int(width)
            self.height = int(height)

    def open(self):
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera.")

        # 높은 해상도를 위해 MJPG 우선
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        def _apply(w, h):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
            aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (aw == int(w) and ah == int(h)), aw, ah

        if self.width and self.height:
            ok, aw, ah = _apply(self.width, self.height)
            if not ok:
                # 적용 실패 시 드라이버 기본값
                aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            ok = False
            aw = ah = 0
            # v4l2-ctl 목록 우선 시도(대해상도부터)
            for w, h in _list_resolutions_v4l2(self.index):
                ok, aw, ah = _apply(w, h)
                if ok:
                    break
            # 목록 실패 시 흔한 해상도 프로빙
            if not ok:
                for w, h in [
                    (7680, 4320), (5120, 2880), (4096, 2160), (3840, 2160),
                    (2560, 1440), (2048, 1536), (1920, 1200), (1920, 1080),
                    (1600, 1200), (1600, 900), (1440, 900),
                    (1366, 768), (1280, 1024), (1280, 800), (1280, 720),
                    (1024, 768), (800, 600), (640, 480)
                ]:
                    ok, aw, ah = _apply(w, h)
                    if ok:
                        break
            if not ok:
                aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.width, self.height = aw, ah

    def read_frame(self):
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Error: Could not read frame.")
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None