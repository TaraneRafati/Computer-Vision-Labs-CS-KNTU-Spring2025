import cv2
import numpy as np

I = cv2.imread('polygons.jpg')
G = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

ret, T = cv2.threshold(G,220,255,cv2.THRESH_BINARY_INV)

nc1,CC1 = cv2.connectedComponents(T)

for k in range(1,nc1):

    Ck = np.zeros(T.shape, dtype=np.float32)
    Ck[CC1 == k] = 1
    Ck = cv2.GaussianBlur(Ck,(5,5),0)

    shape_float = np.float32(Ck)

    window_size = 5
    sobel_kernel_size = 3
    alpha = 0.04
    H = cv2.cornerHarris(shape_float, window_size, sobel_kernel_size, alpha)

    H = H / H.max()
    corner_map = np.uint8(H > 0.01) * 255

    nc2, CC2 = cv2.connectedComponents(corner_map)
    num_corners = nc2 - 1 

    out = cv2.cvtColor(Ck, cv2.COLOR_GRAY2BGR)

    _, _, _, centroids = cv2.connectedComponentsWithStats(corner_map)

    for i in range(1, nc2):
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(Ck,'There are %d vertices!'%(num_corners),(10,30), font, 1,(0,255,0),2)
    
    cv2.imshow('corners',out)
    cv2.waitKey(0) # press any key
