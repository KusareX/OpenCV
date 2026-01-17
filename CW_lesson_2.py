import cv2
import numpy as np

img = cv2.imread("images/black_hole.jpg")

# img = cv2.resize(img, (700, 400))
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.Canny(img, 100, 100)

kernel = np.ones((5, 5), np.uint8)
img = cv2.dilate(img, kernel, iterations=1) #dilate - розширення світлих областей на зобарженні

img = cv2.erode(img, kernel, iterations=1) #ерозія

cv2.imwrite("rendered_images/black_hole.jpg", img)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()