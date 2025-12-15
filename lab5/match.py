import cv2
from matplotlib import pyplot as plt

# Load the grayscale scene image
I = cv2.imread("scene.png", cv2.IMREAD_GRAYSCALE)

# Load the grayscale template image
template = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)

# Compute correlation between template and image
res = cv2.matchTemplate(I, template, cv2.TM_CCOEFF)

# Find the location of the best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# Get template width and height
w, h = template.shape[::-1]

# Define top-left and bottom-right coordinates of the matched region
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# Draw a rectangle around the detected template location
cv2.rectangle(I, top_left, bottom_right, 255, 2)

# Display the correlation map and detected template location
plt.subplot(121), plt.imshow(res, cmap='gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(I, cmap='gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

plt.show()