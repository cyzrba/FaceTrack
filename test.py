import cv2
from deepface import DeepFace
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify
import queue
import threading
import os


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

BATCH_SIZE = 2  # 保持批处理大小


app = Flask(__name__)
yolo_model = YOLO("yolov8n.pt")
print("模型加载完成。")

global_info = {"person_count": 0, "details": []}
track_history = {}  # 维护 ID 的情绪/性别
frame_count = 0  # 帧数计数
analysis_queue = queue.Queue(maxsize=10)  # 异步 DeepFace分析队列


class VideoReader:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG) # RTSP流
        # self.cap = cv2.VideoCapture(0) # 本地摄像头
        self.q = queue.Queue(maxsize=1)  # 队列长度仅为1
        self.stopped = False

        # 启动后台读取线程
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            # 如果队列里有旧帧，直接删掉，只保留最新的
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        #确保取到的是刚从网卡读出来的
        try:
            return self.q.get(timeout=5)  # 设置超时防止死锁
        except queue.Empty:
            return None

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()



def analysis_worker():
    # 字线程 DeepFace分析，更新全局缓存
    global track_history
    COLOR_MAP = {"Woman": (0, 0, 255), "Man": (255, 0, 0)}

    while True:
        try:
            face_region, track_id, _ = analysis_queue.get(timeout=1)

            # DeepFace 分析
            analysis = DeepFace.analyze(face_region, actions=['emotion', 'gender'], enforce_detection=False)

            dom_emotion = max(analysis[0]['emotion'], key=analysis[0]['emotion'].get)
            emotion_conf = analysis[0]['emotion'][dom_emotion]
            dom_gender = analysis[0]['dominant_gender']

            track_history[track_id] = {
                "gender": dom_gender,
                "emotion": dom_emotion,
                "conf": emotion_conf,
                "color": COLOR_MAP.get(dom_gender, (200, 200, 200))
            }
            analysis_queue.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            analysis_queue.task_done()


def generate_frames():
    RTSP_URL = "rtsp://127.0.0.1:8554/live/stream"
    print(f"正在连接 RTSP 流")
    video_reader = VideoReader(RTSP_URL)

    global track_history
    global frame_count

    frame_count = 0
    batch_frames = []

    # 循环读取
    while True:
        # 从读取线程获取最新帧
        frame = video_reader.read()

        if frame is None:
            print("读取流失败")
            continue

        # 批处理缓冲区
        batch_frames.append(frame)

        # 检查是否达到批量大小
        if len(batch_frames) == BATCH_SIZE:

            # YOLO批量推理
            results_list = yolo_model.track(
                batch_frames,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                device=0
            )

            # 遍历批量结果，逐帧处理和推流
            for i, result in enumerate(results_list):
                current_frame = batch_frames[i]
                frame_count += 1

                current_frame_details = []
                person_count = 0

                if result.boxes.id is not None:

                    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    clss = result.boxes.cls.cpu().numpy().astype(int)

                    for box, track_id, cls in zip(boxes, track_ids, clss):
                        if cls != 0: continue

                        person_count += 1
                        x1, y1, x2, y2 = box

                        # 满足则加入任务队列
                        should_analyze = (track_id not in track_history) or (frame_count % 10 == 0)

                        if should_analyze:
                            face_region = current_frame[y1:y2, x1:x2]
                            if face_region.size > 0 and not analysis_queue.full():
                                analysis_queue.put((face_region.copy(), track_id, frame_count))


                        info = track_history.get(track_id,
                                                 {"gender": "?", "emotion": "...", "conf": 0, "color": (255, 255, 255)})

                        current_frame_details.append({
                            "id": int(track_id),
                            "gender": info["gender"],
                            "emotion": info["emotion"],
                            "conf": float(info["conf"]),
                            "box": [int(x1), int(y1), int(x2), int(y2)]
                        })

                        # 绘制
                        cv2.rectangle(current_frame, (x1, y1), (x2, y2), info['color'], 2)
                        label = f"{info['gender']} {info['emotion']}"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(current_frame, (x1, y1 - 20), (x1 + w, y1), info['color'], -1)
                        cv2.putText(current_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                    2)

                # 更新全局数据
                global global_info
                global_info = {
                    "person_count": person_count,
                    "details": current_frame_details
                }

                # 编码推流
                ret, buffer = cv2.imencode('.jpg', current_frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # 清空缓冲区，准备下一批
            batch_frames = []

    video_reader.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/data')
def data():
    global global_info
    return jsonify(global_info)


if __name__ == '__main__':
    worker = threading.Thread(target=analysis_worker, daemon=True) #DeepFace处理线程
    worker.start()
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)