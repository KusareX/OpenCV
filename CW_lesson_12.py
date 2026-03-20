import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)

    if shape == 'circle':
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == 'square':
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == 'triangle':
        points = np.array([[100, 40], [40, 160], [160, 100]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

feats = []
marks = []

colors = {
    'red': (0, 0, 255), 
    'green': (0, 255, 0), 
    'blue': (255, 0, 0), 
    'yellow': (0, 255, 255), 
    'cyan': (255, 255, 0), 
    'magenta': (255, 0, 255), 
    'orange': (0, 165, 255), 
    'purple': (128, 0, 128)}

shapes = ['circle', 'square', 'triangle']

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3]
            feats.append(mean_color)
            marks.append(f'{color_name.capitalize()} {shape.capitalize()}')

feats_train, feats_test, marks_train, marks_test = train_test_split(feats, marks, test_size=0.2)
model = KNeighborsClassifier(n_neighbors=5) 
model.fit(feats_train, marks_train)

#--------------------------------------------------

test_color = (0, 165, 255)
test_shape = 'square'

frames_colors = []
for _ in range(5):
    temp_img = generate_image(test_color, test_shape) 
    frames_colors.append(cv2.mean(temp_img)[:3])
    
smoothed_color = np.mean(frames_colors, axis=0) #зглажування через mean

prediction = model.predict([smoothed_color])
probabilities = model.predict_proba([smoothed_color]) #predict_proba для відсотку голосів сусідів
max_prob = np.max(probabilities) * 100

text = f"{prediction[0]}: {max_prob:.1f}%"
cv2.putText(temp_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

print(text)

cv2.imshow('test', temp_img)
cv2.waitKey(0)
cv2.destroyAllWindows()