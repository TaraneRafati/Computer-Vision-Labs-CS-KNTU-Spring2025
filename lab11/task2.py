import cv2


def apply_transformations(img, scale=1.0, angle=0, brightness=0, contrast=1.0):
    """Apply image transformations"""
    # Apply scaling
    img = cv2.resize(img, None, fx=scale, fy=scale)

    # Apply rotation
    h, w = img.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # Apply brightness/contrast
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    return img


# TODO: Initialize SIFT detector and Create BFMatcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# Load images
scene_img = cv2.imread('scene.jpg')
scene_gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
obj_img = cv2.imread('book1.jpg')
obj_gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)

# TODO: Detect scene features using sift.detectAndCompute()
kp_scene, desc_scene = sift.detectAndCompute(scene_gray, None)

# Transformation tests
transformations = [
    {'name': 'Baseline', 'params': {}},
    {'name': 'Scale 1.2x', 'params': {'scale': 1.2}},
    {'name': 'Rotation 30 degrees', 'params': {'angle': 30}},
    {'name': 'Brightness +40', 'params': {'brightness': 40}},
    {'name': 'Contrast 1.2x', 'params': {'contrast': 1.2}},
    {'name': 'Combined', 'params': {'scale': 0.9, 'angle': 90, 'brightness': 20, 'contrast': 1.1}}
]

for transform in transformations:
    # Apply transformations
    transformed = apply_transformations(obj_gray, **transform['params'])

    # TODO: Detect features on transformed image using sift.detectAndCompute()
    kp_trans, desc_trans = sift.detectAndCompute(transformed, None)

    # TODO: Find two best matches using bf.knnMatch()
    matches = bf.knnMatch(desc_trans, desc_scene, k=2)
    # TODO: Apply Lowe's ratio test to filter out poor matches and keep good matches (use threshold 0.75)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.75 * m2.distance:
            good_matches.append(m1)

    # Visualization
    result_img = cv2.drawMatches(
        cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR), kp_trans,
        scene_img, kp_scene,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.putText(result_img, f"{transform['name']}: {len(good_matches)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Results', result_img)
    cv2.waitKey()

cv2.destroyAllWindows()
