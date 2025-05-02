# %% [1. Introduction to Color Processing]
# Learn color space conversion, color filtering, histogram equalization, and quantization with OpenCV.

# Setup: pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_color_processing_demo():
    # %% [2. Create Synthetic Image]
    # Generate a 300x300 RGB image with a blue-to-red gradient
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(300):
        img[i, :, 0] = i % 255  # Blue gradient
        img[i, :, 2] = 255 - (i % 255)  # Red gradient
    print("Synthetic Image: 300x300 RGB with blue-to-red gradient created")

    # %% [3. Color Space Conversion]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Color Space Conversion: Converted to HSV and Grayscale")

    # %% [4. Color Filtering]
    # Filter blue colors in HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    img_filtered = cv2.bitwise_and(img, img, mask=mask)
    print("Color Filtering: Isolated blue colors")

    # %% [5. Histogram Equalization]
    img_eq = cv2.equalizeHist(img_gray)
    print("Histogram Equalization: Applied to Grayscale image")

    # %% [6. Color Quantization]
    # Reduce colors using K-Means
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    img_quantized = centers[labels.flatten()].reshape(img.shape)
    print(f"Color Quantization: Reduced to {K} colors")

    # %% [7. Visualization]
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 2)
    plt.title("HSV")
    plt.imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
    plt.subplot(2, 3, 3)
    plt.title("Grayscale")
    plt.imshow(img_gray, cmap="gray")
    plt.subplot(2, 3, 4)
    plt.title("Blue Filtered")
    plt.imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 5)
    plt.title("Equalized")
    plt.imshow(img_eq, cmap="gray")
    plt.subplot(2, 3, 6)
    plt.title("Quantized")
    plt.imshow(cv2.cvtColor(img_quantized, cv2.COLOR_BGR2RGB))
    plt.savefig("color_processing_output.png")
    print("Visualization: Color processing saved as color_processing_output.png")

    # %% [8. Interview Scenario: Color Processing]
    """
    Interview Scenario: Color Processing
    Q: Why is HSV useful for color filtering compared to RGB?
    A: HSV separates hue (color type) from saturation and value, making it easier to isolate specific colors.
    Key: Use cv2.inRange in HSV for robust color segmentation.
    Example: cv2.inRange(img_hsv, lower_blue, upper_blue)
    """

# Execute the demo
if __name__ == "__main__":
    run_color_processing_demo()