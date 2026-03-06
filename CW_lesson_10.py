import cv2
import numpy as np
import os

input_folder = 'images'
output_folder = 'output'

formats = ('jpeg', '.png', '.webp', '.jfif', '.tiff')

os.makedirs(output_folder, exist_ok=True)
files = sorted(os.listdir(input_folder))

for file in files:

    path = os.path.join(input_folder, file)
    image = cv2.imread(path)
    if image is None:
        continue
    
    output_path = os.path.join(output_folder, file)
    cv2.imwrite(output_path, image)
    
face_net = cv2.dnn.readNetFromCaffe('data/dnn/deploy.prototxt', 'data/dnn/res10_300x300_ssd_iter_140000.caffemodel')

eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

frame = cv2.imread('images/woman.jpg')
frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
(h, w) = frame.shape[:2]

blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
face_net.setInput(blob)
detections = face_net.forward()

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype('int')
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        face_roi = frame[y1:y2, x1:x2]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x1 + ex, y1 + ey), (x1 + ex + ew, y1 + ey + eh), (0, 255, 0), 2)

cv2.imshow('Face and Eye Detection', frame)
cv2.imwrite('rendered_images/woman_detected.jpg', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()