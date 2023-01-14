import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape) # (60000,)
print(test_images.shape) # (10000, 28, 28)

train_images = train_images.astype('float32') / 255
train_image = train_images.reshape(train_images.shape[0], 28, 28, 1)

test_images = test_images.astype('float32') / 255
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

model = tf.keras.models.Sequential()
model.add(Conv2D( filters=64 , kernel_size=2, padding= 'same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Conv2D( filters=32 , kernel_size=2, padding= 'same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()












