import cv2

img = cv2.imread("images/me.jpg")
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (255, 255, 0), 10)
cv2.putText(img, 'Bogdan Parkhomchuk', (img.shape[1] // 10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2) 
cv2.imshow('key', img)
cv2.waitKey(0)
cv2.destroyAllWindows()