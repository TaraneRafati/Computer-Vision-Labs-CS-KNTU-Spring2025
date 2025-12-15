import numpy as np
import cv2

# Load target and logo image
target_image = cv2.imread('car.jpg')
logo_image = cv2.imread('kntu.jpg')

# Defining destination source_points in the target image
destination_points = np.array([
    (281.85645, 325.7745),
    (478.53232, 329.53046),
    (477.8494,  374.26056),
    (282.8808,  369.8217)
], dtype=np.float32)

# TODO: Define source image corner source_points (the corners of the logo image) using image width and height
h, w = logo_image.shape[:2]
source_points = np.array([
    [0, 0],        
    [w - 1, 0],    
    [w - 1, h - 1],
    [0, h - 1]     
], dtype=np.float32)

# TODO: Compute homography H that maps 'source_points' to 'destination_points' using cv2.getPerspectiveTransform
H = cv2.getPerspectiveTransform(source_points, destination_points)

# TODO: Warp the logo image using the computed homography using cv2.warpPerspective
output_size = (target_image.shape[1], target_image.shape[0])
warped_source = cv2.warpPerspective(logo_image, H, output_size)

# Create a binary mask from the warped logo
gray_warped_source = cv2.cvtColor(warped_source, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray_warped_source, 1, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Extract background and foreground using masks
target_bg = cv2.bitwise_and(target_image, target_image, mask=mask_inv)  
source_fg = cv2.bitwise_and(warped_source, warped_source, mask=mask)

# Blend the warped logo with the background
result = cv2.add(target_bg, source_fg)

# Display the original source image
cv2.imshow('Source Image (Logo)', logo_image)
cv2.waitKey(0)

# Display the original target image
cv2.imshow('Target Image', target_image)
cv2.waitKey(0)

# Show the warped texture alone
cv2.imshow('Warped Source (Intermediate)', warped_source)
cv2.waitKey(0)

# Display the final image with the texture mapped
cv2.imshow('Result (Texture Mapped)', result)
cv2.waitKey(0)
