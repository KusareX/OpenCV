import cv2
import numpy as np

img1 = cv2.imread("images/me.jpg")

img1 = cv2.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2))

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = cv2.Canny(img1, 100, 100)

kernel1 = np.ones((3, 3), np.uint8)

img1 = cv2.dilate(img1, kernel1, iterations=1)
img2 = cv2.erode(img1, kernel1, iterations=1)

cv2.imwrite('rendered_images/me.jpg', img1)
cv2.imshow("image", img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

'-------------------------------------------------------------'

img2 = cv2.imread("images/gmail.jpg")

img2 = cv2.resize(img2, (img2.shape[1], img2.shape[0] // 2))

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.Canny(img2, 100, 100)

kernel2 = np.ones((1, 1), np.uint8)

img2 = cv2.dilate(img2, kernel2, iterations=1)
img2 = cv2.erode(img2, kernel2, iterations=1)

cv2.imwrite('rendered_images/gmail.jpg', img2)
cv2.imshow("image", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()