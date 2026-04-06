import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)

VIDEO_DIR = os.path.join(PROJECT_DIR, 'video')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')

os.makedirs(OUT_DIR, exist_ok=True)

USE_WEBCAM = True

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(PROJECT_DIR, 'video', 'animals.mp4')
    cap = cv2.VideoCapture(VIDEO_PATH)

model = YOLO('yolov8n.pt')

CONF_THRESHOLD = 0.5
RESIZE_WIDTH = 960

CAT_CLASS_ID = 15 # спеціальний клас для котів в YOLO
DOG_CLASS_ID = 16 # спеціальний клас для собак в YOLO

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        
        scale = RESIZE_WIDTH / w
        
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    cat_count = 0
    dog_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == CAT_CLASS_ID:
                cat_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"Cat {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            elif cls == DOG_CLASS_ID:
                dog_count += 1
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Dog {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    total_animals = cat_count + dog_count
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"Cats: {cat_count}", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Dogs: {dog_count}", (20, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Total: {total_animals}", (20, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (840, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)

    cv2.imshow("cotopes", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()