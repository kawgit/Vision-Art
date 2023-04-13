from vae import *
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import random

def load_noise_predictor():
    if os.path.isdir("noise_predictor"):
        noise_predictor = tf.keras.models.load_model('noise_predictor')
    else:
        noisy_encoded_input = keras.Input(shape=(4, 4, 1), name="noisy_encoded")
        label_input = keras.Input(shape=(10,), name="label")

        noisy_encoded_flattened = layers.Flatten()(noisy_encoded_input)
        label_flattened = layers.Flatten()(label_input)

        x = layers.concatenate([noisy_encoded_flattened, label_flattened])
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='tanh')(x)
        y = layers.Reshape((4, 4, 1))(x)

        noise_predictor = keras.Model(inputs=[noisy_encoded_input, label_input], outputs=y, name="noise_predictor")
        noise_predictor.compile(optimizer='adam', loss='mean_squared_error')
    
    # keras.utils.plot_model(noise_predictor, "noise_predictor.png", show_shapes=True)
    return noise_predictor

# labels, noisy_encodeds, noises = generate_noise_dataset()
# noise_predictor = load_noise_predictor()

# for i in range(5):
#     noise_predictor.fit([noisy_encodeds, labels], noises, epochs=5, batch_size=32)
#     noise_predictor.save("noise_predictor")



        # noisy_encoded_input = keras.Input(shape=(4, 4, 1), name="noisy_encoded")
        # label_input = keras.Input(shape=(10,), name="label")

        # label_flattened = layers.Flatten()(label_input)

        # conv = layers.Conv2DTranspose(16, 3, activation="relu", padding="same", strides=1)(noisy_encoded_input)
        # conv = layers.Conv2DTranspose(32, 3, activation="relu", padding="same", strides=1)(conv)
        # conv_flattened = layers.Flatten()(conv)
        # x = layers.concatenate([conv_flattened, label_flattened])
        # x = layers.Dense(32, activation='relu')(x)
        # x = layers.Dense(16, activation='sigmoid')(x)
        # y = layers.Reshape((4, 4, 1))(x)