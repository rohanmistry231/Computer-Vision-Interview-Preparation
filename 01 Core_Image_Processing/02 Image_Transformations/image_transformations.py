# %% [1. Introduction to Image Transformations]
# Learn to resize, crop, rotate, flip, and apply affine/perspective transformations with OpenCV.

# Setup: pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_image_transformations_demo():
    # %% [2. Create Synthetic Image]
    # Generate a 300x300 RGB image with a green rectangle
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[100:200, 100:200] = [0, 255, 0]  # Green square (BGR format)
    print("Synthetic Image: 300x300 RGB with green square created")

    # %% [3. Resizing]
    img_resized = cv2.resize(img, (150, 150), interpolation=cv2.INTER_LINEAR)
    print("Resizing: Image resized to 150x150")

    # %% [4. Cropping]
    img_cropped = img[50:250, 50:250]
    print("Cropping: Image cropped to 200x200")

    # %% [5. Rotation]
    center = (img.shape[1]//2, img.shape[0]//2)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    img_rotated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    print("Rotation: Image rotated by 45 degrees")

    # %% [6. Flipping]
    img_flipped = cv2.flip(img, 1)  # Horizontal flip
    print("Flipping: Image flipped horizontally")

    # %% [7. Translation]
    matrix = np.float32([[1, 0, 50], [0, 1, 50]])  # Shift 50px right, 50px down
    img_translated = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    print("Translation: Image shifted 50px right and down")

    # %% [8. Affine Transformation]
    pts1 = np.float32([[50,50], [200,50], [50,200]])
    pts2 = np.float32([[10,100], [200,50], [100,250]])
    matrix = cv2.getAffineTransform(pts1, pts2)
    img_affine = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
    print("Affine Transformation: Image skewed")

    # %% [9. Perspective Transformation]
    pts1 = np.float32([[50,50], [250,50], [50,250], [250,250]])
    pts2 = np.float32([[0,0], [300,0], [50,300], [250,300]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_perspective = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
    print("Perspective Transformation: Image warped")

    # %% [10. Visualization]
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 4, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 4, 2)
    plt.title("Resized")
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 4, 3)
    plt.title("Cropped")
    plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 4, 4)
    plt.title("Rotated")
    plt.imshow(cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 4, 5)
    plt.title("Flipped")
    plt.imshow(cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 4, 6)
    plt.title("Translated")
    plt.imshow(cv2.cvtColor(img_translated, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 4, 7)
    plt.title("Affine")
    plt.imshow(cv2.cvtColor(img_affine, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 4, 8)
    plt.title("Perspective")
    plt.imshow(cv2.cvtColor(img_perspective, cv2.COLOR_BGR2RGB))
    plt.savefig("image_transformations_output.png")
    print("Visualization: Transformations saved as image_transformations_output.png")

    # %% [11. Interview Scenario: Image Transformations]
    """
    Interview Scenario: Image Transformations
    Q: How do you rotate an image in OpenCV, and what’s the role of the rotation matrix?
    A: Use cv2.getRotationMatrix2D to create a 2D rotation matrix, then apply it with cv2.warpAffine.
    Key: The matrix defines angle, center, and scale for precise rotation.
    Example: cv2.getRotationMatrix2D(center, 45, 1.0)
    """

# Execute the demo
if __name__ == "__main__":
    run_image_transformations_demo()