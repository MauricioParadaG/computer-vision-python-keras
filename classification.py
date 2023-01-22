from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
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

#conv2
model.add(Conv2D(base_filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(w_regulaizer)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#conv3
model.add(Conv2D(2*base_filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(w_regulaizer)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

#conv4
model.add(Conv2D(2*base_filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(w_regulaizer)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

#conv5
model.add(Conv2D(4*base_filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(w_regulaizer)))
model.add(Activation('relu'))

#conv6
model.add(Conv2D(4*base_filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(w_regulaizer)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

#model.summary()

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

batch_size = 32
epochs = 34

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid), verbose=2, shuffle=True)

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

model.evaluate(x_test, y_test) 













