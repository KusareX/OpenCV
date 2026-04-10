import cv2
import os
import yt_dlp
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
# VIDEO_DIR = os.path.join(PROJECT_DIR, 'video')

YOUTUBE_URL = 'https://www.youtube.com/watch?v=M3EYAY2MftI&source_ve_path=NzY3NTg&embeds_referring_euri=https%3A%2F%2Fclassroom.google.com%2F'
MODEL_PATH = 'yolo26n.pt'

track_history = {}
PPM = 8.0

def get_stream_url(url):
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
    }
    print(f"З'єднання з YouTube через yt-dlp...")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        print(f" Помилка: {e}")
        return None

model = YOLO(MODEL_PATH)

stream = get_stream_url(YOUTUBE_URL)
if not stream:
    exit()

cap = cv2.VideoCapture(stream)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

FPS = cap.get(cv2.CAP_PROP_FPS)
if FPS == 0 or FPS != FPS:
    FPS = 30

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.track(frame, classes=[2], conf=0.5, verbose=False)

    if result[0].boxes.id is not None:
        boxes = result[0].boxes.xywh.cpu().numpy()
        track_ids = result[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box

            if track_id in track_history:
                prev_x, prev_y, prev_w, prev_h = track_history[track_id]
                dist_pix = ((x - prev_x) ** 2 + (y - prev_y) ** 2) ** 0.5
                speed_ms = (dist_pix / PPM) * FPS
                speed_kmh = speed_ms * 3.6

                cv2.putText(frame, f'ID: {track_id}, Speed: {speed_kmh:.2f} km/h', (x, y) ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            track_history[track_id] = (x, y, w, h)

    car_frame = result[0].plot()

    frame_count += 1
    cv2.imshow('Live Translation', car_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()