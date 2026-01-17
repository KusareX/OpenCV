import cv2

img = cv2.imread("images/void.jpg")
img = cv2.resize(img, (500, 300))

cv2.imshow("image", img)

vid = cv2.VideoCapture("videos/klychko.mp4")
while True:
    ret, frame = vid.read()
    if not ret:
        break
    frame = cv2.resize(frame, (700, 400))
    
    cv2.imshow("video", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
vid.release()
cv2.destroyAllWindows()