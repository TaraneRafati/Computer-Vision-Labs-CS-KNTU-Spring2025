import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate histogram
def calc_hist(I, levels=256):
    hist = np.zeros(levels)
    I = np.array(I).flatten() 
    for i in I:
        hist[i] += 1
    return hist

# Function to calculate CDF
def calc_cdf(hist):
    cdf = np.zeros_like(hist)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf / cdf[-1]

# Function for histogram equalization
def equalizeHist(I):
    hist = calc_hist(I)
    cdf = calc_cdf(hist)
    mapping = (cdf * 255).astype(np.uint8)
    equalized_image = mapping[I]
    return equalized_image, calc_hist(equalized_image), calc_cdf(calc_hist(equalized_image))

# Load two grayscale images
I1 = cv2.imread("task3p1.png", cv2.IMREAD_GRAYSCALE)
I2 = cv2.imread("task3p2.png", cv2.IMREAD_GRAYSCALE)
I3 = cv2.imread("pasargadae.jpg", cv2.IMREAD_GRAYSCALE)

# Perform histogram equalization
equalized_image1, hist_eq1, cdf_eq1 = equalizeHist(I1)
equalized_image2, hist_eq2, cdf_eq2 = equalizeHist(I2)
equalized_image3, hist_eq3, cdf_eq3 = equalizeHist(I3)

# Compute original histograms and CDFs
hist1, cdf1 = calc_hist(I1), calc_cdf(calc_hist(I1))
hist2, cdf2 = calc_hist(I2), calc_cdf(calc_hist(I2))
hist3, cdf3 = calc_hist(I3), calc_cdf(calc_hist(I3))

# fig, axes = plt.subplots(4, 3)

# axes[0, 0].imshow(I1, cmap='gray')
# axes[0, 0].set_title('Original Image 1')
# axes[0, 0].axis('off')

# axes[1, 0].imshow(equalized_image1, cmap='gray')
# axes[1, 0].set_title('Equalized Image 1')
# axes[1, 0].axis('off')

# axes[2, 0].imshow(I2, cmap='gray')
# axes[2, 0].set_title('Original Image 2')
# axes[2, 0].axis('off')

# axes[3, 0].imshow(equalized_image2, cmap='gray')
# axes[3, 0].set_title('Equalized Image 2')
# axes[3, 0].axis('off')

# axes[0, 1].plot(hist1)
# axes[0, 1].set_title('Histogram (Image 1)')

# axes[1, 1].plot(hist_eq1)
# axes[1, 1].set_title('Histogram (Image 1)')

# axes[2, 1].plot(hist2)
# axes[2, 1].set_title('Histogram (Image 2)')

# axes[3, 1].plot(hist_eq2)
# axes[3, 1].set_title('Histogram (Image 2)')

# axes[0, 2].plot(cdf1)
# axes[0, 2].set_title('CDF (Image 1)')

# axes[1, 2].plot(cdf_eq1)
# axes[1, 2].set_title('CDF (Image 1)')

# axes[2, 2].plot(cdf2)
# axes[2, 2].set_title('CDF (Image 2)')

# axes[3, 2].plot(cdf_eq2)
# axes[3, 2].set_title('CDF (Image 2)')

fig, axes = plt.subplots(2, 3)

axes[0, 0].imshow(I3, cmap='gray')
axes[0, 0].set_title('pasargadea')
axes[0, 0].axis('off')

axes[1, 0].imshow(equalized_image3, cmap='gray')
axes[1, 0].set_title('Equalized Image')
axes[1, 0].axis('off')

axes[0, 1].plot(hist3)
axes[0, 1].set_title('Source Histogram')

axes[1, 1].plot(hist_eq3)
axes[1, 1].set_title('Equalized Histogram')

axes[0, 2].plot(cdf3)
axes[0, 2].set_title('Source CDF')

axes[1, 2].plot(cdf_eq3)
axes[1, 2].set_title('Equalized CDF')


plt.tight_layout()
plt.show()
