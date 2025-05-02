# %% [1. Introduction to Image Filtering]
# Learn blurring, sharpening, edge detection, thresholding, and morphological operations with OpenCV.

# Setup: pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_image_filtering_demo():
    # %% [2. Create Synthetic Image]
    # Generate a 300x300 RGB image with a white rectangle on black
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[100:200, 100:200] = [255, 255, 255]  # White square (BGR format)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Synthetic Image: 300x300 RGB with white square created")

    # %% [3. Blurring]
    img_gaussian = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_median = cv2.medianBlur(img_gray, 5)
    img_bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)
    print("Blurring: Applied Gaussian, Median, and Bilateral filters")

    # %% [4. Sharpening]
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
    img_sharpened = cv2.filter2D(img_gray, -1, kernel)
    print("Sharpening: Applied sharpening filter")

    # %% [5. Edge Detection]
    img_sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel = cv2.convertScaleAbs(img_sobel)
    img_canny = cv2.Canny(img_gray, 100, 200)
    print("Edge Detection: Applied Sobel and Canny")

    # %% [6. Thresholding]
    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    img_adaptive = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    print("Thresholding: Applied Binary and Adaptive")

    # %% [7. Morphological Operations]
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(img_binary, kernel, iterations=1)
    img_erosion = cv2.erode(img_binary, kernel, iterations=1)
    print("Morphological Operations: Applied Dilation and Erosion")

    # %% [8. Visualization]
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 4, 1)
    plt.title("Original")
    plt.imshow(img_gray, cmap="gray")
    plt.subplot(2, 4, 2)
    plt.title("Gaussian Blur")
    plt.imshow(img_gaussian, cmap="gray")
    plt.subplot(2, 4, 3)
    plt.title("Median Blur")
    plt.imshow(img_median, cmap="gray")
    plt.subplot(2, 4, 4)
    plt.title("Sharpened")
    plt.imshow(img_sharpened, cmap="gray")
    plt.subplot(2, 4, 5)
    plt.title("Sobel")
    plt.imshow(img_sobel, cmap="gray")
    plt.subplot(2, 4, 6)
    plt.title("Canny")
    plt.imshow(img_canny, cmap="gray")
    plt.subplot(2, 4, 7)
    plt.title("Binary Threshold")
    plt.imshow(img_binary, cmap="gray")
    plt.subplot(2, 4, 8)
    plt.title("Dilation")
    plt.imshow(img_dilation, cmap="gray")
    plt.savefig("image_filtering_output.png")
    print("Visualization: Filters saved as image_filtering_output.png")

    # %% [9. Interview Scenario: Image Filtering]
    """
    Interview Scenario: Image Filtering
    Q: How does Canny edge detection work, and what are its parameters?
    A: Canny uses Gaussian blur, gradient computation, non-maximum suppression, and hysteresis thresholding.
    Key: Parameters are low/high thresholds for edge strength.
    Example: cv2.Canny(img, 100, 200)
    """

# Execute the demo
if __name__ == "__main__":
    run_image_filtering_demo()