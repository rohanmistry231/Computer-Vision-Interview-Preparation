# %% [1. Introduction to Feature Detection]
# Learn corner detection, keypoint detection, feature matching, and homography with OpenCV.

# Setup: pip install opencv-python opencv-contrib-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_feature_detection_demo():
    # %% [2. Create Synthetic Images]
    # Generate a 300x300 image with a checkerboard pattern
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(0, 300, 50):
        for j in range(0, 300, 50):
            if (i//50 + j//50) % 2 == 0:
                img1[i:i+50, j:j+50] = [255, 255, 255]
    # Create a rotated version
    center = (img1.shape[1]//2, img1.shape[0]//2)
    matrix = cv2.getRotationMatrix2D(center, 10, 1.0)
    img2 = cv2.warpAffine(img1, matrix, (img1.shape[1], img1.shape[0]))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    print("Synthetic Images: Checkerboard and rotated version created")

    # %% [3. Corner Detection (Harris)]
    harris = cv2.cornerHarris(img1_gray, blockSize=2, ksize=3, k=0.04)
    img1_harris = img1.copy()
    img1_harris[harris > 0.01 * harris.max()] = [0, 0, 255]
    print("Harris Corner Detection: Corners marked in red")

    # %% [4. Corner Detection (Shi-Tomasi)]
    corners = cv2.goodFeaturesToTrack(img1_gray, maxCorners=50, qualityLevel=0.01, minDistance=10)
    img1_shi = img1.copy()
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img1_shi, (int(x), int(y)), 3, (0, 255, 0), -1)
    print("Shi-Tomasi Corner Detection: Corners marked in green")

    # %% [5. Keypoint Detection (ORB)]
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)
    img1_orb = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0))
    print(f"ORB Keypoints: {len(keypoints1)} keypoints detected in first image")

    # %% [6. Feature Matching (Brute-Force)]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=2)
    print("Brute-Force Matching: Top 10 matches drawn")

    # %% [7. Homography Estimation]
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = img1_gray.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2_homography = img2.copy()
    cv2.polylines(img2_homography, [np.int32(dst)], True, (0, 0, 255), 3)
    print("Homography Estimation: Transformation outline drawn")

    # %% [8. Visualization]
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title("Original Image 1")
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 2)
    plt.title("Harris Corners")
    plt.imshow(cv2.cvtColor(img1_harris, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 3)
    plt.title("Shi-Tomasi Corners")
    plt.imshow(cv2.cvtColor(img1_shi, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 4)
    plt.title("ORB Keypoints")
    plt.imshow(cv2.cvtColor(img1_orb, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 5)
    plt.title("Feature Matches")
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 6)
    plt.title("Homography")
    plt.imshow(cv2.cvtColor(img2_homography, cv2.COLOR_BGR2RGB))
    plt.savefig("feature_detection_output.png")
    print("Visualization: Feature detection results saved as feature_detection_output.png")

    # %% [9. Interview Scenario: Feature Detection]
    """
    Interview Scenario: Feature Detection
    Q: What’s the difference between SIFT and ORB for keypoint detection?
    A: SIFT is scale-invariant and robust but patented and slower; ORB is fast, open-source, and good for real-time.
    Key: ORB uses binary descriptors, suitable for Hamming distance matching.
    Example: cv2.ORB_create()
    """

# Execute the demo
if __name__ == "__main__":
    run_feature_detection_demo()