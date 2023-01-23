from keras.utils import array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.4, 1.5]
)

img = load_img('./dataset/archive/Churi.jpeg')
x = img_to_array(img)
print(x.shape)  # (800, 600, 3)
x = x.reshape((1,) + x.shape)
print(x.shape)  # (1, 800, 600, 3)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(array_to_img(batch[0]))
    i += 1
    if i % 10 == 0:
        break

train_datagen = datagen.flow_from_directory(
    './dataset/archive/cats_and_dogs/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

print(train_datagen[0][0].shape)  # (32, 150, 150, 3)

array_to_img(train_datagen[0][0][1]).show()


