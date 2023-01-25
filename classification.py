from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)  # (50000, 32, 32, 3)
print(y_train.shape)  # (50000, 1)
print(x_test.shape)  # (10000, 32, 32, 3)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

num_classes = len(np.unique(y_train))  # 10
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Normalize data
mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train - mean) / (std + 1e-7)
x_test = (x_test - mean) / (std + 1e-7)

(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

print('x_train shape:', x_train.shape) # (45000, 32, 32, 3)

print('train' , x_train.shape[0]) # 45000
print('valid' , x_valid.shape[0]) # 5000
print('test' , x_test.shape[0]) # 10000

base_filters = 32
w_regulaizer = 1e-4

model = Sequential()
#conv1
model.add(Conv2D(base_filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(w_regulaizer), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())

#conv2
model.add(Conv2D(base_filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(w_regulaizer)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#conv3
model.add(Conv2D(2*base_filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(w_regulaizer)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

#conv4
model.add(Conv2D(2*base_filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(w_regulaizer)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

#conv5
model.add(Conv2D(4*base_filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(w_regulaizer)))
model.add(Activation('relu'))
model.add(BatchNormalization())

#conv6
model.add(Conv2D(4*base_filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(w_regulaizer)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

#model.summary()

data_generator = ImageDataGenerator(
  rotation_range=15,
  width_shift_range=0.1,
  height_shift_range=0.1,
  horizontal_flip=True,
  vertical_flip=True,
)

model.compile(
  loss='categorical_crossentropy',
  optimizer= Adam(),
  metrics=['accuracy']
)

batch_size = 32
epochs = 50

""" history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid), verbose=2, shuffle=True) """

history = model.fit(
  data_generator.flow(x_train, y_train,  batch_size=epochs),
  callbacks=[ModelCheckpoint(
    'model_Checkpoint.hdf5',
    verbose=1,
    save_best_only=True,
    monitor='val_accuracy',
  )],
  steps_per_epoch=x_train.shape[0] // epochs,
  epochs=epochs,
  validation_data=(x_valid, y_valid),
  verbose=2,
  shuffle=True
)

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

model2 = model
model2.load_weights('model_Checkpoint.hdf5')

model2.evaluate(x_test, y_test) 













