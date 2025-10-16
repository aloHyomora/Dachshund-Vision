import zmq, time, json

def send_results(results, socket):
    message = {"module": "vision_engine", "timestamp": time.time(), "objects": results}
    socket.send_json(message)
