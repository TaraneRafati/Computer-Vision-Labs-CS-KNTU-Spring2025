import numpy as np

A = np.random.rand(200,10)
B = A - A.mean(axis=0)
print(B)