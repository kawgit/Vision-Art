from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2
import random

def load_dataset():
    (trainX, trainY), (testX, testY) = mnist.load_data()

    trainX = trainX.reshape(len(trainX), 28, 28, 1)
    testX = testX.reshape(len(testX), 28, 28, 1)

    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)

    return trainX, trainY, testX, testY

def load_model():
    if os.path.isdir("abc"):
        return tf.keras.models.load_model('abc')
    else:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

trainX, trainY, testX, testY = load_dataset()
model = load_model()

i = random.randint(0, len(testX) - 1)
print(testX[i].shape)
result = model.predict(np.array([testX[i]]))
print(result)
print(np.argmax(result))
img = testX[i].reshape(28, 28)
cv2.imwrite("img.png", img)