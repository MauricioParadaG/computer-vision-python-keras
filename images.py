import numpy as np
import matplotlib.pyplot as plt
from skimage import io

img = io.imread('./dataset/archive/Churi.jpeg')

print(img.shape) # (800, 600, 3)
#print(img) #[[[ 98 112  99]

red = img[:, :, 0]
green = img[:, :, 1]
blue = img[:, :, 2]

print(red.shape) # (800, 600)

plt.imshow(blue.T, cmap='gray')

extra_dim = np.zeros((800, 600))
r =  np.dstack((red, extra_dim, extra_dim)).astype(np.uint8)
g =  np.dstack((extra_dim, green, extra_dim)).astype(np.uint8)
b =  np.dstack((extra_dim, extra_dim, blue)).astype(np.uint8)

img_neg_post = 255 - img
img32 = (img//32) * 32
img128 = (img//128) * 128

#plt.imshow(img32)
plt.imshow(img[10:330, 200:450])
plt.show()


















