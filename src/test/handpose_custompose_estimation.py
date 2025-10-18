import argparse
import time
from typing import List, Tuple, Union

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

# 추가: 제스처 파이프라인
from gestures.gesture_core import GesturePipeline

# 모델 경로
DEFAULT_MODEL_PATH = "models/hand_landmarker.task"

# 손 관절 연결(21점)
HAND_CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)

def draw_hands(img_bgr: np.ndarray,
               hands_xy: List[np.ndarray],
               labels: List[str]) -> None:
    """손 랜드마크와 연결선, 박스를 그립니다."""
    for i, pts in enumerate(hands_xy):
        color = (0, 200, 255)
        # 점
        for x, y in pts:
            cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 255, 255), -1, cv2.LINE_AA)
        # 선
        for a, b in HAND_CONNECTIONS:
            pa = tuple(np.round(pts[a]).astype(int))
            pb = tuple(np.round(pts[b]).astype(int))
            cv2.line(img_bgr, pa, pb, color, 2, cv2.LINE_AA)
        # 박스 + 라벨
        x1, y1 = np.min(pts, axis=0).astype(int)
        x2, y2 = np.max(pts, axis=0).astype(int)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        if i < len(labels) and labels[i]:
            cv2.putText(img_bgr, labels[i], (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

class HandPoseDetector:
    """
    MediaPipe Tasks Hand Landmarker wrapper (VIDEO mode).
    - infer(image_bgr) -> dict with hands_xy, labels, annotated_image
    - Use draw_hands(...) to render onto any image.
    """
    def __init__(self,
                 model_path: str = DEFAULT_MODEL_PATH,
                 num_hands: int = 2,
                 det_conf: float = 0.5,
                 pres_conf: float = 0.5,
                 track_conf: float = 0.5) -> None:
        BaseOptions = python.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        RunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=pres_conf,
            min_tracking_confidence=track_conf,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.start_ms = int(time.monotonic() * 1000)
        self.last_ts = -1

    def _next_timestamp_ms(self) -> int:
        ts_ms = int(time.monotonic() * 1000) - self.start_ms
        if ts_ms <= self.last_ts:
            ts_ms = self.last_ts + 1
        self.last_ts = ts_ms
        return ts_ms

    def infer(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = self._next_timestamp_ms()
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        hands_xy: List[np.ndarray] = []
        labels: List[str] = []
        if result and result.hand_landmarks:
            for i, lm_list in enumerate(result.hand_landmarks):
                pts = np.array([[lm.x * w, lm.y * h] for lm in lm_list], dtype=np.float32)
                hands_xy.append(pts)
                if result.handedness and i < len(result.handedness) and result.handedness[i]:
                    labels.append(result.handedness[i][0].category_name)
                else:
                    labels.append("Hand")

        annotated = image_bgr.copy()
        draw_hands(annotated, hands_xy, labels)

        return {
            "hands_xy": hands_xy,
            "labels": labels,
            "annotated_image": annotated,
        }

def run(source: Union[str, int],
        model_path: str,
        num_hands: int = 2,
        det_conf: float = 0.5,
        pres_conf: float = 0.5,
        track_conf: float = 0.5) -> None:
    # 캡처 열기
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        # MJPG 요청(가능 시 고해상도/고FPS에 유리)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    # Landmarker 생성 (VIDEO 모드)
    BaseOptions = python.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    RunningMode = vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=det_conf,
        min_hand_presence_confidence=pres_conf,
        min_tracking_confidence=track_conf,
    )
    landmarker = HandLandmarker.create_from_options(options)
    pipeline = GesturePipeline()  # 추가

    win = "MediaPipe Hand Landmarker"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass

    # 단조 증가 타임스탬프 기준(초기화 후 고정)
    start_ms = int(time.monotonic() * 1000)
    last_ts = -1
    # FPS 측정용 타이머(별도)
    fps_t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 현재 타임스탬프(단조 증가 보장)
        ts_ms = int(time.monotonic() * 1000) - start_ms
        if ts_ms <= last_ts:
            ts_ms = last_ts + 1
        last_ts = ts_ms

        result = landmarker.detect_for_video(mp_image, ts_ms)

        hands_xy: List[np.ndarray] = []
        hands_world: List[np.ndarray] = []
        labels: List[str] = []
        if result and result.hand_landmarks:
            for i, lm_list in enumerate(result.hand_landmarks):
                pts = np.array([[lm.x * w, lm.y * h] for lm in lm_list], dtype=np.float32)
                hands_xy.append(pts)
                if result.handedness and i < len(result.handedness) and result.handedness[i]:
                    labels.append(result.handedness[i][0].category_name)
                else:
                    labels.append("Hand")

        # world_landmarks가 있으면 3D(m) 좌표 수집
        if result and getattr(result, "hand_world_landmarks", None):
            for i, lm_list in enumerate(result.hand_world_landmarks):
                wpts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
                hands_world.append(wpts)                

        annotated = frame.copy()

        # 추가: 제스처 파이프라인 업데이트 (world 좌표 전달)
        pipe_out = pipeline.update(hands_xy, labels, ts_ms, hands_world=hands_world if hands_world else None)
        overlay_labels = pipe_out["overlay_labels"] if hands_xy else labels
        events = pipe_out["events"] if hands_xy else []

        draw_hands(annotated, hands_xy, overlay_labels)

        # FPS 표시
        dt = time.perf_counter() - fps_t0
        fps = 1.0 / max(1e-6, dt)
        fps_t0 = time.perf_counter()
        cv2.putText(annotated, f"FPS: {fps:.1f}  Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 추가: 제스처 이벤트 로그
    for ev in events:
        print(f"[EVENT] {ev['type']} (hand {ev['hand_id']})")

def parse_args():
    p = argparse.ArgumentParser(description="MediaPipe Tasks Hand Landmarker demo")
    p.add_argument("--source", type=str, default="0", help="웹캠 인덱스(예: 0) 또는 동영상 경로")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="hand_landmarker.task 경로")
    p.add_argument("--num-hands", type=int, default=2)
    p.add_argument("--det-conf", type=float, default=0.5)
    p.add_argument("--pres-conf", type=float, default=0.5)
    p.add_argument("--track-conf", type=float, default=0.5)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    src: Union[str, int] = int(args.source) if args.source.isdigit() else args.source
    run(src, args.model, args.num_hands, args.det_conf, args.pres_conf, args.track_conf)