import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from skimage import io, color

img = io.imread('./dataset/archive/someone.png')
print(img.shape)  # (400, 400, 3)

img_gray = color.rgb2gray(img)
print(img_gray.shape)  # (400, 400)

kernel_vertical = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])  # vertical edge

kernel_horizontal = np.array([[-1, -1, -1],
                              [0, 0, 0],
                              [1, 1, 1]])  # horizontal edge
kernel = np. array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])  # identity


convolution_image = ndimage.convolve(img_gray, kernel)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_gray, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(convolution_image, cmap='gray')
axes[1].set_title('Convolution')
axes[1].axis('off')

plt.show()
