import cv2
import numpy as np

I = cv2.imread('karimi.jpg',0)

tx = 0
ty = 0

th =  20 # angle of rotation (degrees)
th *= np.pi / 180 # convert to radians

s = 2

M = np.array([[s * np.cos(th),s * -np.sin(th),tx],
              [s * np.sin(th), s * np.cos(th),ty]])


w, h = I.shape[1], I.shape[0]

corners = np.array([
    [0, 0],
    [w, 0],
    [0, h],
    [w, h]
], dtype=np.float32)

transformed_corners = cv2.transform(np.array([corners]), M)[0]

xmin, ymin = transformed_corners.min(axis=0)
xmax, ymax = transformed_corners.max(axis=0)

new_w = int(np.ceil(xmax - xmin))
new_h = int(np.ceil(ymax - ymin))

M[0, 2] -= xmin
M[1, 2] -= ymin

J = cv2.warpAffine(I,M, (new_w, new_h) )

cv2.imshow('I',I)
cv2.waitKey(0)

cv2.imshow('J',J)
cv2.waitKey(0)
