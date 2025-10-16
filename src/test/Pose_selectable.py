import argparse
import time
from typing import Optional, Union

import cv2
import numpy as np
# 추가: 상위(src) 모듈 import 위해 경로 추가
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from handpose_estimation import HandPoseDetector, draw_hands, DEFAULT_MODEL_PATH as DEFAULT_HAND_MODEL
from pose_estimation import PoseDetector  # YOLOv8 pose
from vision_engine import VisionEngine    # 객체 감지 엔진
from camera_input import CameraInput      # 추가: 최대 해상도 카메라 래퍼

DEFAULT_BODY_MODEL = "models/yolov8n-pose.pt"
DEFAULT_OBJ_MODEL = "models/yolov8n.pt"


def open_capture(source: Union[int, str]) -> Optional[cv2.VideoCapture]:
    # 파일/네트워크 소스만 처리 (웹캠은 CameraInput 사용)
    if isinstance(source, int):
        return None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    return cap


def main():
    parser = argparse.ArgumentParser(description="Selectable Hand/Body/Object Pose via OpenCV trackbars (max camera resolution)")
    parser.add_argument("--source", type=str, default="0", help="웹캠 인덱스(예: 0) 또는 동영상 경로")
    parser.add_argument("--hand-model", type=str, default=DEFAULT_HAND_MODEL, help="hand_landmarker.task 경로")
    parser.add_argument("--body-model", type=str, default=DEFAULT_BODY_MODEL, help="YOLOv8 Pose 모델 경로")
    parser.add_argument("--obj-model", type=str, default=DEFAULT_OBJ_MODEL, help="YOLOv8 Detect 모델 경로")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO pose confidence")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO pose NMS IoU")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO pose inference size")
    parser.add_argument("--obj-conf", type=float, default=0.25, help="YOLO detect confidence")
    parser.add_argument("--obj-iou", type=float, default=0.45, help="YOLO detect NMS IoU")
    args = parser.parse_args()

    source: Union[int, str] = int(args.source) if args.source.isdigit() else args.source

    hand_detector: Optional[HandPoseDetector] = None
    body_detector: Optional[PoseDetector] = None
    object_detector: Optional[VisionEngine] = None

    # 윈도우 생성
    win = "Pose Selectable (cv2)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_NORMAL)

    # Trackbars (토글 0/1)
    cv2.createTrackbar("Hand", win, 1, 1, lambda v: None)
    cv2.createTrackbar("Body", win, 1, 1, lambda v: None)
    cv2.createTrackbar("Object", win, 0, 1, lambda v: None)

    fps_t0 = time.perf_counter()

    if isinstance(source, int):
        # CameraInput로 최대 해상도 자동 적용
        cam = CameraInput(index=source, width=0, height=0)
        try:
            cam.open()
        except Exception as e:
            print(f"[ERROR] Cannot open camera {source}: {e}")
            return

        window_name = f"{win} | cam{source} {cam.width}x{cam.height}"
        try:
            while True:
                try:
                    frame = cam.read_frame()
                except Exception as e:
                    print(f"[ERROR] {e}")
                    break

                overlay = frame.copy()

                hand_on = cv2.getTrackbarPos("Hand", win) == 1
                body_on = cv2.getTrackbarPos("Body", win) == 1
                object_on = cv2.getTrackbarPos("Object", win) == 1

                status_msgs = []

                # BODY
                if body_on:
                    if body_detector is None:
                        body_detector = PoseDetector(
                            model_path=args.body_model, conf=args.conf, iou=args.iou, imgsz=args.imgsz
                        )
                    out_body = body_detector.infer(frame, person_only=True)
                    overlay = out_body["annotated_image"]
                    status_msgs.append("Body:on")
                else:
                    status_msgs.append("Body:off")

                # HAND
                if hand_on:
                    if hand_detector is None:
                        hand_detector = HandPoseDetector(model_path=args.hand_model)
                    out_hand = hand_detector.infer(frame)
                    draw_hands(overlay, out_hand["hands_xy"], out_hand["labels"])
                    status_msgs.append("Hand:on")
                else:
                    status_msgs.append("Hand:off")

                # OBJECT
                if object_on:
                    if object_detector is None:
                        object_detector = VisionEngine(
                            model_path=args.obj_model, device="cpu", conf=args.obj_conf, iou=args.obj_iou
                        )
                    dets = object_detector.infer(frame)
                    overlay = object_detector.draw(overlay, dets)
                    status_msgs.append("Obj:on")
                else:
                    status_msgs.append("Obj:off")

                # FPS 표기
                dt = time.perf_counter() - fps_t0
                fps = 1.0 / max(1e-6, dt)
                fps_t0 = time.perf_counter()
                cv2.putText(
                    overlay, f"{window_name} | FPS: {fps:.1f} | " + " ".join(status_msgs) + " | q/ESC",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                )

                cv2.imshow(win, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()
    else:
        # 파일 소스
        cap = open_capture(source)
        if cap is None:
            print(f"[ERROR] Cannot open source: {args.source}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        window_name = f"{win} | file {w}x{h}"

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[INFO] Stream ended or failed")
                    break

                overlay = frame.copy()

                hand_on = cv2.getTrackbarPos("Hand", win) == 1
                body_on = cv2.getTrackbarPos("Body", win) == 1
                object_on = cv2.getTrackbarPos("Object", win) == 1

                status_msgs = []

                # BODY
                if body_on:
                    if body_detector is None:
                        body_detector = PoseDetector(
                            model_path=args.body_model, conf=args.conf, iou=args.iou, imgsz=args.imgsz
                        )
                    out_body = body_detector.infer(frame, person_only=True)
                    overlay = out_body["annotated_image"]
                    status_msgs.append("Body:on")
                else:
                    status_msgs.append("Body:off")

                # HAND
                if hand_on:
                    if hand_detector is None:
                        hand_detector = HandPoseDetector(model_path=args.hand_model)
                    out_hand = hand_detector.infer(frame)
                    draw_hands(overlay, out_hand["hands_xy"], out_hand["labels"])
                    status_msgs.append("Hand:on")
                else:
                    status_msgs.append("Hand:off")

                # OBJECT
                if object_on:
                    if object_detector is None:
                        object_detector = VisionEngine(
                            model_path=args.obj_model, device="cpu", conf=args.obj_conf, iou=args.obj_iou
                        )
                    dets = object_detector.infer(frame)
                    overlay = object_detector.draw(overlay, dets)
                    status_msgs.append("Obj:on")
                else:
                    status_msgs.append("Obj:off")

                # FPS 표기
                dt = time.perf_counter() - fps_t0
                fps = 1.0 / max(1e-6, dt)
                fps_t0 = time.perf_counter()
                cv2.putText(
                    overlay, f"{window_name} | FPS: {fps:.1f} | " + " ".join(status_msgs) + " | q/ESC",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                )

                cv2.imshow(win, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()