from vae import *
from noise import *
from data import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import random
import os
import numpy as np
import time

encoder, decoder, autoencoder = load_vae()
noise_predictor = load_noise_predictor()

def train_autoencoder():
    (train_imgs, train_labels), (test_imgs, test_labels) = read_basic()
    global autoencoder
    global encoder
    global decoder
    
    while True:
        autoencoder.fit(train_imgs, train_imgs, epochs=1, batch_size=32)
        encoder.save('encoder')
        decoder.save('decoder')

def generate_noise_dataset():
    (train_encodes, train_labels), (test_encodes, test_labels) = read_encoded()

    noisy_encodes = []
    noises = []
    labels = []

    for i, encode in enumerate(train_encodes):
        noise = np.zeros(ENCODE_SHAPE)

        print(train_labels[i])
        display_encode(encode)
        input("")

        for j in range(10):
            noisy_encode = np.add(encode, noise)

            noisy_encodes.append(noisy_encode)
            labels.append(train_labels[i])
            noises.append(noise)

            noise = -np.subtract(noisy_encode, np.clip(np.add(noisy_encode, np.random.uniform(low = -.4, high = .4, size=ENCODE_SHAPE)), 0, 1))
        
    # temp = list(zip(train_labels, noisy_encodes, noises))
    # random.shuffle(temp)
    # labels, noisy_encodes, noises = zip(*temp)
    
    # labels = np.array(labels)
    # noisy_encodes = np.array(noisy_encodes)
    # noises = np.array(noises)

    return np.array(labels), np.array(noisy_encodes), np.array(noises)

def train_noise_predictor():
    labels, noisy_encodes, noises = generate_noise_dataset()

    print(labels.shape)
    print(noisy_encodes.shape)
    print(noises.shape)

    while True:
        noise_predictor.fit(x=[noisy_encodes, labels], y=noises, epochs=10, batch_size=128)
        noise_predictor.save('noise_predictor')

def display_encode(encode):
    img = np.clip(np.rint(encode * 255.0), 0, 255)
    cv2.imwrite("encoded.png", img)

    decode = decoder.predict(np.array([encode]))[0]
    img = np.clip(np.rint(decode * 255.0), 0, 255)
    cv2.imwrite("decoded.png", img)

def gen_img_of(prompt):
    label = keras.utils.to_categorical(prompt, num_classes=10)
    noisy_encode = np.random.uniform(low = .2, high = .8, size=ENCODE_SHAPE)
    print(noisy_encode)
    for i in range(30):
        print(prompt)
        noise_prediction = noise_predictor.predict([np.array([noisy_encode]), np.array([label])])[0]
        # print(noise_prediction)
        noisy_encode = np.clip(np.subtract(noisy_encode, noise_prediction * (.9 ** i)), 0, 1)
        display_encode(noisy_encode)

# clear_dir('./assets/basic/test')
# clear_dir('./assets/basic/train')
# clear_dir('./assets/encoded/test')
# clear_dir('./assets/encoded/train')

train_autoencoder()

# train_noise_predictor()

for i in range(10):
    print(i)
    gen_img_of(i)
    input("")


# next up: http://vision.stanford.edu/aditya86/ImageNetDogs/