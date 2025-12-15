import numpy as np
import cv2

I = cv2.imread('img1.bmp', cv2.IMREAD_GRAYSCALE)

n,C = cv2.connectedComponents(I)

output = np.zeros((C.shape[0], C.shape[1], 3), dtype=np.uint8)

np.random.seed(42)
colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(n)]
colors[0] = [0, 0, 0]  

for i in range(n):
    output[C == i] = colors[i]

font = cv2.FONT_HERSHEY_SIMPLEX 

cv2.putText(output,'There are %d connected components!'%(n-1),(20,40), font, 1,(255,255,255),2)

cv2.imshow('Colored Components', output)
cv2.waitKey(0)

