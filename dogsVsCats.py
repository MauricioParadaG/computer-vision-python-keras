import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './dataset/archive/cats_and_dogs/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    './dataset/archive/cats_and_dogs/validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

checkpoint = ModelCheckpoint(
    'model_dogsvscats.hdf5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
)

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=0.0001),
    metrics=['accuracy']
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // 32,
    epochs=90,
    validation_data=validation_generator,
    validation_steps=1000 // 32,
    callbacks=[checkpoint],
    verbose=1
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

test_generator = test_datagen.flow_from_directory(
    './dataset/archive/cats_and_dogs/test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

model2 = model
model2.load_weights('model_dogsvscats.hdf5')

model2.evaluate(test_generator)


