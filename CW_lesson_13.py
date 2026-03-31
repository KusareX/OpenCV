import pandas as pd #pandas для роботи з scv файлами
import numpy as np
import tensorflow as tf
from tensorflow import keras #keras для роботи з api
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('data/figures.csv')
print(df.head())

encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

X = df[['area', 'perimeter', 'corners']]
y = df['label_enc']

model = keras.Sequential([layers.Dense(8, activation='relu', input_shape=(3,)),
                          layers.Dense(8, activation='relu'), #другий шар називається прихований
                          layers.Dense(3, activation='softmax')
                          ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, y, epochs=500, verbose=0)

plt.plot(history.history['loss'], label='Втрати')
plt.plot(history.history['accuracy'], label='Точність')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Процес навчання моделі')
plt.legend()
plt.show()

test = np.array([(42, 26, 4)])

preds = model.predict(test)
print(f'Ймовірність кожного класу: {preds*100}%')
print(f'Результат {encoder.inverse_transform(np.argmax(preds, axis=1))}')