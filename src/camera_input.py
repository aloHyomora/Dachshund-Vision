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
            req_w, req_h = int(self.width), int(self.height)
            ok, aw, ah = _apply(req_w, req_h)
            if not ok:
                # 1) v4l2가 보고한 지원 해상도 중 요청치 이하에서 최댓값 선택
                for w, h in _list_resolutions_v4l2(self.index):
                    if w <= req_w and h <= req_h:
                        ok, aw, ah = _apply(w, h)
                        if ok:
                            break
            if not ok:
                # 2) 흔한 해상도 후보들 중 요청치 이하에서 시도(상대적으로 작은 쪽부터)
                for w, h in [
                    (1280, 720), (1024, 768), (960, 540), (800, 600), (640, 480)
                ]:
                    if w <= req_w and h <= req_h:
                        ok, aw, ah = _apply(w, h)
                        if ok:
                            break
            if not ok:
                # 3) 마지막 수단: 드라이버 기본값 유지
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