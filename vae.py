import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import random

def load_encoder():
    if os.path.isdir("encoder"):
        encoder = tf.keras.models.load_model('encoder')
    else:
        encoder = keras.Sequential([
            layers.Reshape((28, 28, 1)),
            layers.Conv2D(64, 5, activation="relu"),
            layers.Conv2D(32, 4, activation="relu"),
            layers.MaxPooling2D(3),
            layers.Conv2D(16, 2, activation="sigmoid"),
            layers.GlobalMaxPooling2D(),
            layers.Reshape((6, 6, 1)),
        ], name="encoder")

        encoder.build(input_shape=(None, 28, 28, 1))
    keras.utils.plot_model(encoder, "encoder.png", show_shapes=True)
    return encoder


def load_decoder():
    if os.path.isdir("decoder"):
        decoder = tf.keras.models.load_model('decoder')
    else:
        decoder = keras.Sequential([
            layers.Reshape((4, 4, 1)),
            layers.Conv2DTranspose(16, 2, activation="relu"),
            layers.Conv2DTranspose(32, 3, activation="relu"),
            layers.UpSampling2D(3),
            layers.Conv2DTranspose(16, 4, activation="relu"),
            layers.Conv2DTranspose(1, 5, activation="sigmoid"),
            layers.Reshape((28, 28, 1)),
        ], name="decoder")
        decoder.build(input_shape=(None, 4, 4, 1))
    keras.utils.plot_model(decoder, "decoder.png", show_shapes=True)
    return decoder


def make_autoencoder(encoder, decoder):
    encoder_input = keras.Input(shape=(28, 28, 1), name="img") 
    encoder_output = encoder(encoder_input)
    decoder_output = decoder(encoder_output)
    autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    keras.utils.plot_model(autoencoder, "autoencoder.png", show_shapes=True)
    return autoencoder

def load_vae():
    encoder = load_encoder()
    decoder = load_decoder()
    autoencoder = make_autoencoder(encoder, decoder)
    return encoder, decoder, autoencoder

# encoder, decoder, autoencoder = load_vae()

# trainX, trainY, testX, testY = load_dataset()

# def go(img):
#     cv2.imwrite("original_img.png", arr_to_img(img))

#     global encoder
#     encoded = encoder.predict(np.array([img]))
#     encoded_img = arr_to_img(encoded, 4)
#     cv2.imwrite("encoded_img.png", encoded_img)

#     global decoder
#     decoded = decoder.predict(encoded)
#     decoded_img = arr_to_img(decoded)
#     cv2.imwrite("decoded_img.png", decoded_img)

# while True:
#     i = random.randint(0, len(testX) - 1)
#     go(testX[i])

#     # encoded_img = np.random.randint(low=0, high=255, size=(4, 4))
#     # cv2.imwrite("encoded_img.png", encoded_img)
#     # encoded = encoded_img / 255.0
#     # decoded = decoder.predict(np.array([encoded]))
#     # decoded_img = arr_to_img(decoded)
#     # cv2.imwrite("decoded_img.png", decoded_img)
    
#     input("")

# autoencoder.fit(trainX, trainX, epochs=5, batch_size=32)
# decoder.save('decoder')
# encoder.save('encoder')

# go(trainX[0])
