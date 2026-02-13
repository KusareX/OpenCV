import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_red_1 = np.array([0, 100, 100])
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([160, 100, 100])
upper_red_2 = np.array([180, 255, 255])

points = []

while True:
    ret, frame = cap.read() #ret це boolean значення що відповідає за зчитання кадру
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1) #inRange для створення маски
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    mask = cv2.bitwise_or(mask_1, mask_2) #bitwise_and об'єднує дві маски
    
    countours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in countours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 2)
            
            mnt = cv2.moments(cnt)
            if mnt['m00'] != 0:
                cx = int(mnt['m10'] / mnt['m00'])
                cy = int(mnt['m01'] / mnt['m00'])
                
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                points.append((cx, cy))
                
                for i in range(1, len(points)):
                    if points[i-1] is None or points[i] is None:
                        continue
                    
                    cv2.line(frame, points[i-1], points[i], (255, 0, 0), 2)
                
    cv2.imshow('frame', frame)   
    cv2.imshow('mask', mask) 
      
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()