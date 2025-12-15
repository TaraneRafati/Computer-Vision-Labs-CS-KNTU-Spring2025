import cv2
import numpy as np

def non_max_suppression(H, window_size=3):

    suppressed = np.zeros_like(H)
    offset = window_size // 2

    # TODO: Loop over every pixel (except borders) and suppress non-maximums
    for i in range(offset, H.shape[0] - offset):
        for j in range(offset, H.shape[1] - offset):
            window = H[i-offset:i+offset+1, j-offset:j+offset+1]
            # TODO: Keep only local maximum
            if H[i, j] == np.max(window):
                suppressed[i, j] = H[i, j]
    
    return suppressed

# Load image
I = cv2.imread('square.jpg')
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
G = np.float32(G)

# Harris parameters
block_size = 2
sobel_ksize = 3
alpha = 0.04

# Compute Harris response
H = cv2.cornerHarris(G, block_size, sobel_ksize, alpha)
H = H / H.max()  # normalize


#Apply Non-Maximum Suppression
# TODO: Call your non_max_suppression function with appropriate window size
H_nms = non_max_suppression(H, window_size=5)

# Threshold the result
# TODO: Threshold to get binary corner map (uint8)
corner_map = np.uint8(H_nms > 0.01) * 255


#Color corners (1px)
I[corner_map != 0] = [0, 0, 255]


# Find connected components and their centroids
# TODO: Use connectedComponentsWithStats
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(corner_map)


# TODO: Loop through all detected components (excluding background)
for i in range(1, num_labels):
    x, y = int(centroids[i][0]), int(centroids[i][1])
    cv2.circle(I, (x, y), 3, (0, 255, 0), -1)
    print(f"Corner {i}: (x={x}, y={y})")

# Show and annotate
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(I, f'Total Corners: {num_labels - 1}', (20, 40), font, 1, (0, 0, 255), 2)

cv2.imshow('Corners with NMS', I)
cv2.waitKey(0)
cv2.destroyAllWindows()
