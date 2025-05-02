# %% [1. Introduction to Template Matching]
# Learn single and multi-object template matching with OpenCV.

# Setup: pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_template_matching_demo():
    # %% [2. Create Synthetic Image and Template]
    # Generate a 300x300 image with multiple white squares
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[50:100, 50:100] = [255, 255, 255]  # Square 1
    img[150:200, 150:200] = [255, 255, 255]  # Square 2
    img[250:300, 50:100] = [255, 255, 255]  # Square 3
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a 50x50 template (white square)
    template = np.ones((50, 50), dtype=np.uint8) * 255
    print("Synthetic Image: 300x300 with three white squares; Template: 50x50 white square")

    # %% [3. Single Object Matching]
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    img_single = img.copy()
    top_left = max_loc
    bottom_right = (top_left[0] + 50, top_left[1] + 50)
    cv2.rectangle(img_single, top_left, bottom_right, (0, 0, 255), 2)
    print(f"Single Object Matching: Best match at {max_loc}, score={max_val:.2f}")

    # %% [4. Multi-Object Matching]
    threshold = 0.8
    loc = np.where(result >= threshold)
    img_multi = img.copy()
    for pt in zip(*loc[::-1]):
        bottom_right = (pt[0] + 50, pt[1] + 50)
        cv2.rectangle(img_multi, pt, bottom_right, (0, 255, 0), 2)
    print(f"Multi-Object Matching: {len(loc[0])} matches found with threshold={threshold}")

    # %% [5. Visualization]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title("Single Match")
    plt.imshow(cv2.cvtColor(img_single, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 3)
    plt.title("Multi Match")
    plt.imshow(cv2.cvtColor(img_multi, cv2.COLOR_BGR2RGB))
    plt.savefig("template_matching_output.png")
    print("Visualization: Template matching results saved as template_matching_output.png")

    # %% [6. Interview Scenario: Template Matching]
    """
    Interview Scenario: Template Matching
    Q: How does template matching work, and what are its limitations?
    A: Template matching slides a template over an image, computing similarity (e.g., normalized correlation).
    Key: Limited to fixed-scale, rotation-invariant templates; sensitive to lighting changes.
    Example: cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    """

# Execute the demo
if __name__ == "__main__":
    run_template_matching_demo()