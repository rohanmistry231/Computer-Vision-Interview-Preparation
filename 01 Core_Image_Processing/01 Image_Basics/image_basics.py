# %% [1. Introduction to Image Basics]
# Learn to read/write images, manipulate pixels, and explore color spaces with OpenCV.

# Setup: pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_image_basics_demo():
    # %% [2. Create Synthetic Image]
    # Generate a 300x300 RGB image with a red rectangle
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[100:200, 100:200] = [0, 0, 255]  # Red square (BGR format)
    print("Synthetic Image: 300x300 RGB with red square created")

    # %% [3. Reading/Writing Images]
    cv2.imwrite("synthetic_image.png", img)
    img_read = cv2.imread("synthetic_image.png")
    print("Image Read/Written: synthetic_image.png")

    # %% [4. Color Spaces]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(f"Color Spaces: Converted to Grayscale and HSV")

    # %% [5. Pixel Manipulation]
    img[50:100, 50:100] = [255, 0, 0]  # Add blue square
    print("Pixel Manipulation: Added blue square at (50:100, 50:100)")

    # %% [6. Image Properties]
    height, width, channels = img.shape
    print(f"Image Properties: Height={height}, Width={width}, Channels={channels}")

    # %% [7. Visualization]
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 3, 1)
    plt.title("Original (BGR)")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title("Grayscale")
    plt.imshow(img_gray, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("HSV")
    plt.imshow(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
    plt.savefig("image_basics_output.png")
    print("Visualization: Images saved as image_basics_output.png")

    # %% [8. Interview Scenario: Image Basics]
    """
    Interview Scenario: Image Basics
    Q: What’s the difference between RGB and HSV color spaces?
    A: RGB defines colors by red, green, blue intensities; HSV uses hue, saturation, value for intuitive color manipulation.
    Key: HSV is better for color-based segmentation (e.g., filtering specific hues).
    Example: cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    """

# Execute the demo
if __name__ == "__main__":
    run_image_basics_demo()