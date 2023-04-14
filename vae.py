import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import random
from data import *

DECODE_WIDTH = 28
DECODE_HEIGHT = 28
DECODE_CHANNELS = 1
DECODE_SHAPE = (DECODE_WIDTH, DECODE_HEIGHT, DECODE_CHANNELS)

ENCODE_WIDTH = 12
ENCODE_HEIGHT = 12
ENCODE_CHANNELS = 3
ENCODE_SHAPE = (ENCODE_WIDTH, ENCODE_HEIGHT, ENCODE_CHANNELS)

def load_encoder():
    if os.path.isdir("encoder"):
        encoder = tf.keras.models.load_model('encoder')
    else:
        print("AYOAYO")
        encoder = keras.Sequential([
            layers.Reshape(DECODE_SHAPE),
            layers.Conv2D(16, 5, activation="relu"),
            layers.Conv2D(16, 5, activation="sigmoid"),
            layers.Conv2D(16, 5, activation="sigmoid"),
            layers.Conv2D(5, 3, activation="sigmoid"),
            layers.Conv2D(3, 3, activation="sigmoid"),
            layers.Reshape(ENCODE_SHAPE),
        ], name="encoder")

        encoder.build(input_shape=(None, *DECODE_SHAPE))
    # keras.utils.plot_model(encoder, "encoder.png", show_shapes=True)
    return encoder


def load_decoder():
    if os.path.isdir("decoder"):
        decoder = tf.keras.models.load_model('decoder')
    else:
        decoder = keras.Sequential([
            layers.Reshape(ENCODE_SHAPE),
            layers.Conv2DTranspose(16, 3, activation="sigmoid"),
            layers.Conv2DTranspose(16, 3, activation="sigmoid"),
            layers.Conv2DTranspose(16, 5, activation="sigmoid"),
            layers.Conv2DTranspose(5, 5, activation="sigmoid"),
            layers.Conv2DTranspose(1, 5, activation="sigmoid"),
            layers.Reshape(DECODE_SHAPE),
        ], name="decoder")
        decoder.build(input_shape=(None, *ENCODE_SHAPE))
    # keras.utils.plot_model(decoder, "decoder.png", show_shapes=True)
    return decoder


def make_autoencoder(encoder, decoder):
    encoder_input = keras.Input(shape=DECODE_SHAPE, name="img") 
    encoder_output = encoder(encoder_input)
    decoder_output = decoder(encoder_output)
    autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    # keras.utils.plot_model(autoencoder, "autoencoder.png", show_shapes=True)
    return autoencoder

def load_vae():
    encoder = load_encoder()
    decoder = load_decoder()
    autoencoder = make_autoencoder(encoder, decoder)
    return encoder, decoder, autoencoder

def fetch_encoded():
    encoder = load_encoder()
    (train_imgs, train_labels), (test_imgs, test_labels) = read_basic()
    train_encodes = encoder.predict(train_imgs)
    test_encodes = encoder.predict(test_imgs)
    return (train_encodes, train_labels), (test_encodes, test_labels)

def write_encoded():
    encoder = load_encoder()
    (train_imgs, train_labels), (test_imgs, test_labels) = read_basic()
    print(train_imgs[0])
    train_encodes = encoder.predict(train_imgs)
    print(train_encodes[0])
    exit()
    test_encodes = encoder.predict(test_imgs)
    write_pairs('./assets/encoded/train', train_encodes, train_labels)
    write_pairs('./assets/encoded/test', test_encodes, test_labels)

def read_encoded():
    return fetch_encoded()
    if len(os.listdir('./assets/encoded/train')) == 0 or len(os.listdir('./assets/encoded/test')) == 0:
        write_encoded()
    return read_pairs('./assets/encoded/train', ENCODE_SHAPE), read_pairs('./assets/encoded/test', ENCODE_SHAPE)