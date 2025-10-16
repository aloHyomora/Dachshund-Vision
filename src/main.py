import argparse
import time
import zmq
import cv2
from vision_engine import VisionEngine
from camera_input import CameraInput
from zmq_publisher import send_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)   # 0: 자동 최대 해상도
    parser.add_argument("--height", type=int, default=0)  # 0: 자동 최대 해상도
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--no-show", dest="no_show", action="store_true", help="창 표시 없이 퍼블리시만")
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.connect(args.endpoint)

    cam = CameraInput(index=args.index, width=args.width, height=args.height)
    cam.open()  # MJPG + 자동 최대 해상도 적용됨

    engine = VisionEngine(model_path=args.model, device="cpu", conf=args.conf, iou=args.iou)

    window_name = f"YOLOv8 (CPU) cam{args.index} {cam.width}x{cam.height} - q/ESC to quit"
    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    period = 1.0 / max(1e-6, args.hz)
    try:
        while True:
            t0 = time.time()
            frame = cam.read_frame()

            detections = engine.infer(frame)
            send_results(detections, sock)

            if not args.no_show:
                vis = engine.draw(frame, detections)
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        sock.close(0)
        ctx.term()

if __name__ == "__main__":
    main()
