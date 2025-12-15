import cv2
from matplotlib import pyplot as plt
import numpy as np

def classify_image(I):
    hist = cv2.calcHist([I], [0], None, [256], [0, 256])
    mean_intensity = np.mean(I)
    std_dev = np.std(I)
    dark_pixels = np.sum(hist[:50]) / I.size
    bright_pixels = np.sum(hist[200:]) / I.size

    category = ""

    if mean_intensity < 50 or dark_pixels > 0.5:
        category += "Underexposed "
    elif mean_intensity > 200 or bright_pixels > 0.5:
        category += "Overexposed "

    if std_dev < 50:
        category += "Low Contrast "
    elif 50 <= mean_intensity <= 200 and 0.2 <= dark_pixels <= 0.4 and 0.2 <= bright_pixels <= 0.4:
        category += "Well-balanced "
    return category

image_path = 'crayfish.jpg'
I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
category = classify_image(I)
print(f"The image is classified as: {category}")

fig, axes = plt.subplots(1, 2)
plt.suptitle(category)

axes[0].imshow(I, 'gray')
axes[0].set_title('Image')
axes[0].axis('off')

axes[1].hist(I.ravel(), 256, [0, 256])
axes[1].set_title('Histogram')

plt.tight_layout()
plt.show()
