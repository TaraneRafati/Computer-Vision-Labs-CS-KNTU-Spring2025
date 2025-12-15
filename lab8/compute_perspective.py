import cv2
import numpy as np

I1 = cv2.imread('farshchian1.jpg')
I2 = cv2.imread('farshchian2.jpg')

points1 = np.array([(82,14),
                    (242,17),
                    (241, 207),
                    (81, 206)]).astype(np.float32)

points2 = np.array([(46,75),
                    (196,61),
                    (220,227),
                    (76,251)]).astype(np.float32)
 
for i in range(4):
    cv2.circle(I1, (int(points1[i,0]), int(points1[i,1])), 3, (0, 0, 255), 2)
    cv2.circle(I2, (int(points2[i,0]), int(points2[i,1])), 3, (0, 0, 255), 2)


# compute homography from point correspondences
H = cv2.getPerspectiveTransform(points1, points2)

output_size = (I2.shape[1], I2.shape[0])
J = cv2.warpPerspective(I1,H,  output_size)

# I2 to I1
H_inv = np.linalg.inv(H)
J_reversed = cv2.warpPerspective(I2, H_inv, (I1.shape[1], I1.shape[0]))

cv2.imshow('I1',I1)
cv2.waitKey(0)

cv2.imshow('I2',I2)
cv2.waitKey(0)

cv2.imshow('J',J)
cv2.waitKey(0)

cv2.imshow('J_inv',J_reversed)
cv2.waitKey(0)