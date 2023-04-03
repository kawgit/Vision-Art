from vae import *
from noise import *
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

def clear_dir(dir_path):
    for file in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, file))

def write_pairs(dest_path, imgs, labels):
    imgs = np.rint(imgs * 255.0)
    assert(len(labels[0]) == 10)
    for i, img in enumerate(imgs):
        id = random.randint(0, 999999)
        name = str(id).rjust(6, '0') + '_' + str(np.argmax(labels[i]))
        path = dest_path + "/" + name + '.png'
        # print(path)
        cv2.imwrite(path, img)

def read_pairs(src_path, w=28):
    start = time.time()

    files = os.listdir(src_path)
    imgs = np.zeros((len(files), w, w, 1))
    labels = np.zeros((len(files), 10), dtype='i')

    print("files found:", len(files))

    for i, file in enumerate(files):
        imgs[i] = np.array(cv2.imread(os.path.join(src_path, file), cv2.IMREAD_GRAYSCALE)).reshape(w, w, 1) / 255.0
        labels[i][int(file.split('_')[1].split('.')[0])] = 1

    print("elapsed: ", time.time() - start)

    return imgs, labels

def write_basic():
    (train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_imgs = train_imgs / 255.0
    test_imgs = test_imgs / 255.0
    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)
    clear_dir('./assets/basic/train')
    clear_dir('./assets/basic/test')
    write_pairs('./assets/basic/train', train_imgs, train_labels)
    write_pairs('./assets/basic/test', test_imgs, test_labels)

def read_basic():
    if len(os.listdir('./assets/basic/train')) == 0 or len(os.listdir('./assets/basic/test')) == 0:
        write_basic()
    return read_pairs('./assets/basic/train'), read_pairs('./assets/basic/test')

def write_encoded():
    encoder = load_encoder()
    (train_imgs, train_labels), (test_imgs, test_labels) = read_basic()
    train_encodes = encoder.predict(train_imgs)
    test_encodes = encoder.predict(test_imgs)
    write_pairs('./assets/encoded/train', train_encodes, train_labels)
    write_pairs('./assets/encoded/test', test_encodes, test_labels)

def read_encoded():
    if len(os.listdir('./assets/encoded/train')) == 0 or len(os.listdir('./assets/encoded/test')) == 0:
        write_encoded()
    return read_pairs('./assets/encoded/train', 4), read_pairs('./assets/encoded/test', 4)

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
        noise = np.zeros((4, 4, 1))

        # print(train_labels[i])
        # display_encode(encode)
        # input("")

        for j in range(10):
            noisy_encode = np.add(encode, noise)

            noisy_encodes.append(noisy_encode)
            labels.append(train_labels[i])
            noises.append(noise)

            noise = -np.subtract(noisy_encode, np.clip(np.add(noisy_encode, np.random.uniform(low = -.4, high = .4, size=(4, 4, 1))), 0, 1))
        
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
    noisy_encode = np.random.uniform(low = .2, high = .8, size=(4, 4, 1))
    print(noisy_encode)
    for i in range(30):
        print(prompt)
        noise_prediction = noise_predictor.predict([np.array([noisy_encode]), np.array([label])])[0]
        # print(noise_prediction)
        noisy_encode = np.clip(np.subtract(noisy_encode, noise_prediction * (.9 ** i)), 0, 1)
        display_encode(noisy_encode)

clear_dir('./assets/basic/test')
clear_dir('./assets/basic/train')
clear_dir('./assets/encoded/test')
clear_dir('./assets/encoded/train')

# train_autoencoder()

# train_noise_predictor()

for i in range(10):
    print(i)
    gen_img_of(i)
    input("")


# next up: http://vision.stanford.edu/aditya86/ImageNetDogs/