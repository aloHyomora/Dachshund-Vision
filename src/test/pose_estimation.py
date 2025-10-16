from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional, Tuple

# 부모(src) 디렉터리를 sys.path에 추가
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from camera_input import CameraInput  # 추가: 안정적 카메라 입력 래퍼


class PoseDetector:
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> None:
        """
        YOLOv8 Pose로 사람 박스 + 관절을 함께 추론.
        """
        self.model = YOLO(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    @torch.inference_mode()
    def infer(
        self, image_bgr: np.ndarray, person_only: bool = True
    ) -> Dict[str, Any]:
        """
        입력: BGR 이미지(OpenCV). 출력: 박스, 스코어, 클래스, 키포인트, 주석이미지.
        """
        # Ultralytics는 BGR/RGB 모두 처리 가능. 그대로 넣어도 OK.
        results = self.model(
            image_bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False
        )[0]

        # 바운딩 박스
        boxes = results.boxes  # Boxes object
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.detach().cpu().numpy()  # (N, 4)
            scores = boxes.conf.detach().cpu().numpy()  # (N,)
            cls = boxes.cls.detach().cpu().numpy().astype(int)  # (N,)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            cls = np.empty((0,), dtype=np.int32)

        # 키포인트 (x, y, conf)
        kpts_obj = results.keypoints  # Keypoints object
        if kpts_obj is not None and kpts_obj.data is not None and len(kpts_obj.data) > 0:
            # data shape: (N, K, 3) -> x, y, conf
            kpts = kpts_obj.data.detach().cpu().numpy()  # (N, K, 3)
            kpts_xy = kpts[..., :2]  # (N, K, 2)
            kpts_conf = kpts[..., 2]  # (N, K)
        else:
            kpts_xy = np.empty((0, 17, 2), dtype=np.float32)  # 기본 COCO 17 keypoints 가정
            kpts_conf = np.empty((0, 17), dtype=np.float32)

        # 필요 시 사람 클래스(0)만 필터링
        if person_only and cls.size > 0:
            keep = cls == 0
            xyxy = xyxy[keep]
            scores = scores[keep]
            cls = cls[keep]
            if kpts_xy.shape[0] == keep.shape[0]:
                kpts_xy = kpts_xy[keep]
                kpts_conf = kpts_conf[keep]

        # 주석(박스+스켈레톤) 렌더링
        annotated = results.plot()  # boxes + skeleton 모두 그림

        return {
            "boxes_xyxy": xyxy,            # (N, 4)
            "scores": scores,              # (N,)
            "class_ids": cls,              # (N,)
            "keypoints_xy": kpts_xy,       # (N, K, 2)
            "keypoints_conf": kpts_conf,   # (N, K)
            "annotated_image": annotated,  # BGR
            "names": results.names,        # 클래스 ID -> 이름
            "orig_shape": results.orig_shape,
        }


def run_realtime(
    source: str | int = 0,
    model_path: str = "yolov8n-pose.pt",
    device: Optional[str] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    cam_width: int = 0,    # 추가: 카메라 요청 해상도(0=자동)
    cam_height: int = 0,   # 추가: 카메라 요청 해상도(0=자동)
) -> None:
    detector = PoseDetector(model_path=model_path, device=device, conf=conf, iou=iou, imgsz=imgsz)

    win_name = "YOLOv8 Pose - Person boxes + keypoints"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_NORMAL)

    # 카메라인 경우: CameraInput 사용 / 파일 경로인 경우: 기존 VideoCapture 사용
    if isinstance(source, int):
        try:
            cam = CameraInput(index=source, width=cam_width, height=cam_height)
            cam.open()
        except Exception as e:
            print(f"[ERROR] Cannot open camera {source}: {e}", file=sys.stderr)
            return

        try:
            while True:
                frame = cam.read_frame()
                out = detector.infer(frame, person_only=True)
                annotated = out["annotated_image"]

                cv2.putText(
                    annotated, "Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.imshow(win_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {source}", file=sys.stderr)
            return

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                out = detector.infer(frame, person_only=True)
                annotated = out["annotated_image"]

                cv2.putText(
                    annotated, "Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.imshow(win_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Pose: 사람 박스 + 관절 동시 추론")
    parser.add_argument("--source", type=str, default="0", help="웹캠 인덱스(예: 0) 또는 파일 경로")
    parser.add_argument("--model", type=str, default="models/yolov8n-pose.pt", help="Pose 모델 경로")
    parser.add_argument("--device", type=str, default=None, help="cpu 또는 cuda")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--cam-width", type=int, default=0, help="카메라 요청 너비(0=자동)")   # 추가
    parser.add_argument("--cam-height", type=int, default=0, help="카메라 요청 높이(0=자동)")  # 추가
    args = parser.parse_args()

    # source가 숫자 문자열이면 웹캠 인덱스로 처리
    source: str | int
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    run_realtime(
        source=source,
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        cam_width=args.cam_width,     # 전달
        cam_height=args.cam_height,   # 전달
    )


if __name__ == "__main__":
    main()