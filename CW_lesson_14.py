import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(__file__)

TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'train')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'test')

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR, image_size=(128, 128), batch_size=32,
    label_mode="categorical"
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR, image_size=(128, 128), batch_size=32,
    label_mode="categorical"
)

model = models.Sequential()

model.add(layers.Rescaling(1./255, input_shape=(128, 128, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu')) #activaion='relu' визначає максимальне значення, перетворюючи від'ємні значення на нуль
model.add(layers.MaxPooling2D((2,2))) #MaxPooling зменшує картинку, прибираючи шум (зайві елементи)

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128 ,(3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.compile(
    optimizer='adam',
    loss= 'categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds, epochs=20, validation_data=test_ds)

test_photo = os.path.join(TEST_DIR, 'images', 'test_photo.jpg')

if os.path.exists(test_photo):
    img = image.load_img(test_photo, target_size=(128, 128))
    img_array = image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    class_name = sorted(os.listdir(TRAIN_DIR))

    result_index = np.argmax(predictions[0])

    print(f'Результат: {class_name[result_index]}')
