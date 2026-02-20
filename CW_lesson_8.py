import cv2
import numpy as np

frame = cv2.imread('images/candies.jpg')
frame = cv2.resize(frame, (800, 600))
frame = cv2.flip(frame, 1)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_red_1 = np.array([0, 100, 100])
upper_red_1 = np.array([10, 255, 255])
lower_red_2 = np.array([160, 100, 100])
upper_red_2 = np.array([180, 255, 255])

lower_blue_1 = np.array([100, 100, 100])
upper_blue_1 = np.array([130, 255, 255])

lower_yellow_1 = np.array([20, 100, 100])
upper_yellow_1 = np.array([30, 255, 255])

mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

mask_blue = cv2.inRange(hsv, lower_blue_1, upper_blue_1)

mask_yellow = cv2.inRange(hsv, lower_yellow_1, upper_yellow_1)

contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#мені здається тут краще було б зробити через функцію, щоб не дублювати те саме три рази, але я не дуже розумію як саме, тому вже як є

for cnt in contours_red:
    area = cv2.contourArea(cnt)
    if area > 1500:
        cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
        mnt = cv2.moments(cnt)
        if mnt['m00'] != 0:
            cx = int(mnt['m10'] / mnt['m00'])
            cy = int(mnt['m01'] / mnt['m00'])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, 'Red', (cx-50, cy-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

for cnt in contours_blue:
    area = cv2.contourArea(cnt)
    if area > 1500:
        cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 2)
        mnt = cv2.moments(cnt)
        if mnt['m00'] != 0:
            cx = int(mnt['m10'] / mnt['m00'])
            cy = int(mnt['m01'] / mnt['m00'])
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(frame, 'Blue', (cx-10, cy-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
for cnt in contours_yellow:
    area = cv2.contourArea(cnt)
    if area > 1500:
        cv2.drawContours(frame, [cnt], -1, (0, 255, 255), 2)
        mnt = cv2.moments(cnt)
        if mnt['m00'] != 0:
            cx = int(mnt['m10'] / mnt['m00'])
            cy = int(mnt['m01'] / mnt['m00'])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(frame, 'Yellow', (cx-40, cy-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
cv2.imwrite("rendered_images/candies_detected.jpg", frame)
cv2.imshow('frame', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()