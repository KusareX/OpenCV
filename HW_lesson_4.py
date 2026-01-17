import cv2

img = cv2.imread('images/fridge.jpg')
img = cv2.resize(img, (600, 800))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 50, 150)

contours, _= cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

objects_found = 0

for contour in contours:
    area = cv2.contourArea(contour)
    
    if area > 100:        
        x, y, w, h = cv2.boundingRect(contour)
        objects_found += 1
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

print(f'Кількість знайдених об\'єктів: {objects_found}')
cv2.imshow('img', img)
cv2.imwrite('rendered_images/fridge_detected.jpg', img)

cv2.waitKey(0)
cv2.destroyAllWindows()