import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
import cv2
import time

# all images should be stored on drives with shape=(W, H, C) with chars [0, 255]
# all images should be stored in memory with shape=(W, H, C) with floats [0, 1]

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

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

def read_pairs(src_path, shape=(28, 28, 1)):
    start = time.time()

    files = os.listdir(src_path)
    imgs = np.zeros((len(files), *shape))
    labels = np.zeros((len(files), 10), dtype='i')

    print("files found:", len(files))

    for i, file in enumerate(files):
        imgs[i] = np.array(cv2.imread(os.path.join(src_path, file), cv2.IMREAD_GRAYSCALE)).reshape(shape) / 255.0
        labels[i][int(file.split('_')[1].split('.')[0])] = 1

        if i % 100 == 0:
            print(i, "/", len(files))

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