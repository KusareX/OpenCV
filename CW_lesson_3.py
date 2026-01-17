import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)

# img[100:400, 100:400] = 255, 0, 0  # blue square
# img[:] = 0, 255, 0  # full screen green square

cv2.rectangle(img, (100, 100), (400, 400), (255, 0, 0), -1)  # blue square
cv2.circle(img, (256, 256), 100, (0, 255, 0), -1)  # green circle
cv2.line(img, (0, 0), (512, 512), (0, 0, 255), thickness=3)  # red line
cv2.line(img, (0, img.shape[0] // 2), (0, img.shape[0] // 2), (255, 255, 0), thickness=3)  # cyan line
cv2.putText(img, 'OpenCV', (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)  # white text

cv2.imshow('primitives', img)
cv2.waitKey(0)
cv2.destroyAllWindows()