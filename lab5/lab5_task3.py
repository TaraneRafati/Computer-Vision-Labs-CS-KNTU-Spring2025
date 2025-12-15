import cv2
import numpy as np
import matplotlib.pyplot as plt

def adaptive_median_filter(img, kernel_size=3):
    """Applies an adaptive median filter to reduce noise while preserving edges."""
    return cv2.medianBlur(img, kernel_size)

def max_directional_difference(img):
    """Computes the maximum directional difference in the RGB space."""
    diff = np.zeros(img.shape[:2], dtype=np.float32)
    for i in range(3):  # Iterate over R, G, B channels
        #Compute Sobel in x direction in the selected channel
        sobelx = np.abs(cv2.Sobel(img[:, :, i], cv2.CV_64F, 1, 0))
        # Compute Sobel in y direction in the selected channel
        sobely = np.abs(cv2.Sobel(img[:, :, i], cv2.CV_64F, 0, 1))
        diff = np.maximum(diff, np.sqrt(sobelx**2 + sobely**2))
    return diff

def threshold_edge_detection(diff, threshold_ratio=0.3):
    """Applies a threshold to the detected edges."""
    threshold = threshold_ratio * np.max(diff)
    return (diff > threshold).astype(np.uint8) * 255

# Load image
image = cv2.imread("pepper.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply adaptive median filter
filtered_img = adaptive_median_filter(image, kernel_size=3)

# Compute max directional difference
edge_map = max_directional_difference(image)

# Apply thresholding
final_edges = threshold_edge_detection(edge_map)

# Display results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(edge_map, cmap="gray")
ax[1].set_title("Max Directional Difference")
ax[1].axis("off")

ax[2].imshow(final_edges, cmap="gray")
ax[2].set_title("Final Edge Map")
ax[2].axis("off")

plt.show()




