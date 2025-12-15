import numpy as np
import cv2

def classify_transformation(matrix):
    if matrix.shape != (3, 3):
        print("Unsupported matrix shape. Only 3x3 matrices are allowed.")
        return

    #[0 0 0]
    #[0 0 0]
    #[0 0 0]
    A = matrix[:2, :2]
    t = matrix[:2, 2:3]
    bottom_row = matrix[2, :]

    ATA = A.T @ A

    print("Transformation type: ", end='')

    # TODO: Print the appropriate Transformation type in the console
    if np.allclose(bottom_row, [0, 0, 1], atol=1e-4): 
        if np.allclose(A, np.eye(2)) and not np.allclose(t, 0):
            print("Translation")
        elif np.allclose(ATA, np.eye(2), atol=1e-2):
            print("Rigid (Rotation + Translation)")
        elif np.allclose(ATA[0, 0], ATA[1, 1], atol=1e-2) and np.allclose(ATA[0, 1], 0, atol=1e-2):
            print("Similarity (Rotation + Uniform Scaling + Translation)")
        else:
            print("Affine")
    else:
        print("Projective (Homography)")


# Pure translation
T = np.array([
    [1, 0, 100],
    [0, 1, 50],
    [0, 0, 1]
])
classify_transformation(T)

# Rigid (rotation + translation)
theta = np.deg2rad(30)
R = np.array([
    [np.cos(theta), -np.sin(theta), 20],
    [np.sin(theta),  np.cos(theta), 10],
    [0, 0, 1]
])
classify_transformation(R)

# Similarity
M = np.array([
    [2 * np.cos(0.4), -2 * np.sin(0.4), 30],
    [2 * np.sin(0.4), 2 * np.cos(0.4), 40],
    [0, 0, 1]
])
classify_transformation(M)

# Affine (shear)
A = np.array([
    [1.2, 0.3, 10],
    [0.1, 1.1, 15],
    [0, 0, 1]
])
classify_transformation(A)

# Projective
P = np.array([
    [1, 0.2, 30],
    [0.1, 1, 40],
    [0.0005, 0.0002, 1]
])
classify_transformation(P)



