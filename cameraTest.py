import cv2
import sys
import os
import re
import subprocess

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
    return sorted(sizes, key=lambda wh: (wh[0] * wh[1], wh[0]), reverse=True)

def _try_set_resolution(cap, w: int, h: int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (aw == int(w) and ah == int(h)), aw, ah

def open_camera(index=0, width=0, height=0):
    # Open the camera
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit()

    # Try MJPG to unlock higher resolutions on many UVC cameras
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # If width/height not specified, try to use the maximum supported
    if width <= 0 or height <= 0:
        selected_w = selected_h = None

        # Prefer exact list from v4l2-ctl if available
        for w, h in _list_resolutions_v4l2(index):
            ok, aw, ah = _try_set_resolution(cap, w, h)
            if ok:
                selected_w, selected_h = aw, ah
                break

        # Fallback: probe common resolutions from highest to lowest
        if selected_w is None:
            for w, h in [
                (7680, 4320), (5120, 2880), (4096, 2160), (3840, 2160),
                (2560, 1440), (2048, 1536), (1920, 1200), (1920, 1080),
                (1600, 1200), (1600, 900), (1440, 900),
                (1366, 768), (1280, 1024), (1280, 800), (1280, 720),
                (1024, 768), (800, 600), (640, 480)
            ]:
                ok, aw, ah = _try_set_resolution(cap, w, h)
                if ok:
                    selected_w, selected_h = aw, ah
                    break

        # Final fallback: accept whatever the driver provides
        if selected_w is None:
            selected_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            selected_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        width, height = selected_w, selected_h
    else:
        # Honor requested size, but reflect the actual applied size
        _, aw, ah = _try_set_resolution(cap, width, height)
        width, height = aw, ah

    window_name = f"Webcam {index} - {width}x{height} - press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # 'q' or ESC
            break
        # 창을 닫기 버튼으로 닫은 경우도 종료
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    open_camera(cam_idx)