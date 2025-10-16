# YOLOv8 + CameraInput: CPU 모드 추론, 결과(클래스, 좌표, 확률) 출력
from ultralytics import YOLO
from camera_input import CameraInput
import cv2
import argparse
import sys

class VisionEngine:
    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu", conf: float = 0.25, iou: float = 0.45):
        self.model = YOLO(model_path)
        # CPU 테스트 모드 (추후 TensorRT 엔진으로 교체 예정)
        self.model.to(device)
        self.names = self.model.names
        self.conf = conf
        self.iou = iou

    def infer(self, frame):
        # 단일 프레임 추론
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "class_id": cls_id,
                "class_name": self.names.get(cls_id, str(cls_id)),
                "bbox_xyxy": [x1, y1, x2, y2],
                "score": conf
            })
        return detections

    def draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            label = f'{det["class_name"]} {det["score"]:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + max(60, len(label) * 8), y1), (0, 200, 0), -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLOv8 모델 가중치")
    parser.add_argument("--index", type=int, default=0, help="카메라 인덱스")
    # 0=자동 최대 해상도 (camera_input에서 v4l2-ctl 탐색 + MJPG 적용)
    parser.add_argument("--width", type=int, default=0, help="0 for auto max")
    parser.add_argument("--height", type=int, default=0, help="0 for auto max")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--no-show", dest="no_show", action="store_true", help="창 표시 없이 콘솔 출력만")
    args = parser.parse_args()

    cam = CameraInput(index=args.index, width=args.width, height=args.height)
    try:
        cam.open()
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

    engine = VisionEngine(model_path=args.model, device="cpu", conf=args.conf, iou=args.iou)

    # 실제 적용된 해상도 표시
    window_name = f"YOLOv8 (CPU) cam{args.index} {cam.width}x{cam.height} - q/ESC to quit"
    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    try:
        while True:
            frame = cam.read_frame()
            detections = engine.infer(frame)

            # 콘솔 로그: 클래스, 좌표, 확률
            if detections:
                for d in detections:
                    print(f'{d["class_name"]}\t{d["bbox_xyxy"]}\t{d["score"]:.3f}')
            else:
                print("no detections")

            if not args.no_show:
                vis = engine.draw(frame, detections)
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # q or ESC
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cam.release()
        if not args.no_show:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

