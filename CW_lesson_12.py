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
    'blue': (255, 0, 0)
}

shapes = ['circle', 'square', 'triangle']

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(30):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3]
            preds = [mean_color[0], mean_color[1], mean_color[2]]

            feats.append(preds)
            marks.append(f'{color_name.capitalize()} {shape}')

feats_train, feats_test, marks_train, marks_test = train_test_split(feats, marks, test_size=0.3, stratify=marks)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(feats_train, marks_train)

accuracy = model.score(feats_test, marks_test)

print(f'Точність моделі: {100 * accuracy:.2f}%')

test_img = generate_image((222, 21, 10), 'square')
mean_color = cv2.mean(test_img)[:3]
preds = model.predict([mean_color])

print(f'Передбачення: {preds}')

cv2.imshow('test', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()