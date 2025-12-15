import cv2
import matplotlib.pyplot as plt

image_path = 'inversion.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
inverted_image = 255 - image

fig, axes = plt.subplots(1, 2)

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(inverted_image, cmap='gray')
axes[1].set_title('Inverted Image')
axes[1].axis('off')

plt.show()
