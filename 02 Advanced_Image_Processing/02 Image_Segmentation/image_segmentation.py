# %% [1. Introduction to Image Segmentation]
# Learn contour detection, watershed, GrabCut, and K-Means clustering with OpenCV.

# Setup: pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_image_segmentation_demo():
    # %% [2. Create Synthetic Image]
    # Generate a 300x300 image with overlapping shapes
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 100), 50, (255, 0, 0), -1)  # Blue circle
    cv2.rectangle(img, (100, 150, 100, 100), (0, 255, 0), -1)  # Green rectangle
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Synthetic Image: 300x300 with blue circle and green rectangle created")

    # %% [3. Contour Detection]
    _, thresh = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 2)
    print(f"Contour Detection: {len(contours)} contours found")

    # %% [4. Watershed Algorithm]
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(thresh, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_watershed = img.copy()
    markers = cv2.watershed(img_watershed, markers)
    img_watershed[markers == -1] = [255, 0, 0]
    print("Watershed Algorithm: Segmented overlapping regions")

    # %% [5. GrabCut]
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 200, 200)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_grabcut = img * mask2[:, :, np.newaxis]
    print("GrabCut: Foreground segmented with rectangular initialization")

    # %% [6. K-Means Clustering]
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    img_kmeans = centers[labels.flatten()].reshape(img.shape)
    print(f"K-Means Clustering: Segmented into {K} colors")

    # %% [7. Visualization]
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 2)
    plt.title("Contours")
    plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 3)
    plt.title("Watershed")
    plt.imshow(cv2.cvtColor(img_watershed, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 4)
    plt.title("GrabCut")
    plt.imshow(cv2.cvtColor(img_grabcut, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 5)
    plt.title("K-Means")
    plt.imshow(cv2.cvtColor(img_kmeans, cv2.COLOR_BGR2RGB))
    plt.savefig("image_segmentation_output.png")
    print("Visualization: Segmentation results saved as image_segmentation_output.png")

    # %% [8. Interview Scenario: Image Segmentation]
    """
    Interview Scenario: Image Segmentation
    Q: How does GrabCut work for image segmentation?
    A: GrabCut uses graph cuts to separate foreground and background, initialized with a rectangle or mask.
    Key: Iteratively refines segmentation using Gaussian Mixture Models.
    Example: cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5)
    """

# Execute the demo
if __name__ == "__main__":
    run_image_segmentation_demo()