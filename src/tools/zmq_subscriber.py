import argparse, json, time, zmq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="tcp://*:5555", help="SUB가 bind할 주소(퍼블리셔는 127.0.0.1:5555로 connect)")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--connect", action="store_true", help="bind 대신 connect 하고 싶을 때 사용")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")  # 모든 토픽 수신
    if args.connect:
        sub.connect(args.endpoint)
    else:
        sub.bind(args.endpoint)

    # PUB/SUB slow joiner 방지
    time.sleep(0.2)

    print(f"[INFO] Listening on {args.endpoint} (pretty={args.pretty})")
    while True:
        parts = sub.recv_multipart()
        topic = None
        payload = parts[-1]
        if len(parts) > 1:
            try:
                topic = parts[0].decode("utf-8", "ignore")
            except Exception:
                topic = None

        try:
            text = payload.decode("utf-8")
            obj = json.loads(text)
            if args.pretty:
                from pprint import pprint
                if topic: print(f"[{topic}]")
                pprint(obj)
            else:
                if topic:
                    print(f"[{topic}] {obj}")
                else:
                    print(obj)
        except Exception:
            # JSON이 아니면 원문 출력
            if topic:
                print(f"[{topic}] {payload!r}")
            else:
                print(repr(payload))

if __name__ == "__main__":
    main()