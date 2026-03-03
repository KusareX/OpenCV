import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_smile.xml')

face_net = cv2.dnn.readNetFromCaffe('data/dnn/deploy.prototxt', 'data/dnn/res10_300x300_ssd_iter_140000.caffemodel')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:

        break

#____________________DNN____________________
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0)) #адаптування зображення під нейронку

    face_net.setInput(blob) #пропускання зображення через нейронку
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (x1, y1, x2, y2) = box.astype('int')

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('dnn', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#____________________Cascades____________________
#     grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) #детекутємо обличчя
#
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #малюємо прямокуктнк навколо обличчя
#
#         roi_gray = grey[y:y + h, x:x + w]
#         roi_color = frame[y:y + h, x:x + w]
#
#         eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30)) #детектуємо очі
#
#         smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15)) #детектуємо очі
#
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2) #малюємо прямокуктнк навколо очей
#
#         for (ix, iy, iw, ih) in smile:
#             cv2.rectangle(roi_color, (ix, iy), (ix + iw, iy + ih), (0, 0, 255), 2)  # малюємо прямокуктнк навколо посмішки
#
#     cv2.putText(
#         frame, f'Faces detected: {len(faces)}.',
#         (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1
#     )
#
#     cv2.imshow('Haar face tracking', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()