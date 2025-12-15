import numpy as np
import cv2

############ Load image and convert to grayscale
I = cv2.imread('coins.jpg')
G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

############ Step 0: Invert the grayscale image so that coins become white and background becomes black
############ This helps with better thresholding and connected component detection
G = cv2.bitwise_not(G)
cv2.imshow('Grayscale', G)
cv2.waitKey(0)  # press any key to continue...

############ Step 1: Try applying a threshold
############ Replace 127 with a value you find appropriate
ret, T = cv2.threshold(G, 35, 255, cv2.THRESH_BINARY)
cv2.imshow('Thresholded', T)
cv2.waitKey(0)

############ Step 2: Try improving segmentation using morphological operations
############ You can try erosion to separate connected objects or remove noise
############ You may also use dilation to fill holes inside coins
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
T = cv2.erode(T, kernel)
cv2.imshow('After Erosion', T)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
T = cv2.dilate(T, kernel)
cv2.imshow('After Dilation', T)
cv2.waitKey(0)

############ Step 3: Count connected components
n, C = cv2.connectedComponents(T)
print("Number of connected components (including background): ", n)
print("Estimated number of coins: ", n - 1)

############ Optional: annotate the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(T, 'There are %d coins!' % (n - 1), (20, 40), font, 1, 255, 2)
cv2.imshow('Result', T)
cv2.waitKey(0)
