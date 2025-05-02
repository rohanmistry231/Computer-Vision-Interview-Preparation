# %% [1. Introduction to Video Analysis]
# Learn frame differencing, background subtraction, motion detection, and stabilization with OpenCV.

# Setup: pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_video_analysis_demo():
    # %% [2. Create Synthetic Video Frames]
    # Generate 10 frames (300x300) with a moving white square
    frames = []
    for i in range(10):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        x = 100 + i * 10  # Move 10px right per frame
        frame[x:x+50, 100:150] = [255, 255, 255]  # White square
        frames.append(frame)
    frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    print("Synthetic Video: 10 frames with moving white square created")

    # %% [3. Frame Differencing]
    diff = cv2.absdiff(frames_gray[4], frames_gray[5])
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    print("Frame Differencing: Motion detected between frames 5 and 6")

    # %% [4. Background Subtraction (MOG2)]
    bg_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2()
    mog2_mask = bg_subtractor_mog2.apply(frames[5])
    print("Background Subtraction: MOG2 applied to frame 6")

    # %% [5. Background Subtraction (KNN)]
    bg_subtractor_knn = cv2.createBackgroundSubtractorKNN()
    knn_mask = bg_subtractor_knn.apply(frames[5])
    print("Background Subtraction: KNN applied to frame 6")

    # %% [6. Motion Detection]
    contours, _ = cv2.findContours(mog2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_motion = frames[5].copy()
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_motion, (x, y), (x+w, y+h), (0, 0, 255), 2)
    print(f"Motion Detection: {len(contours)} moving objects detected")

    # %% [7. Video Stabilization (Simplified)]
    # Simulate stabilization by aligning frame 6 to frame 5 using translation
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(frames_gray[4], None)
    kp2, des2 = orb.detectAndCompute(frames_gray[5], None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    img_stabilized = cv2.warpAffine(frames[5], M, (300, 300))
    print("Video Stabilization: Frame 6 aligned to frame 5")

    # %% [8. Visualization]
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.title("Frame 6")
    plt.imshow(cv2.cvtColor(frames[5], cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 2)
    plt.title("Frame Diff")
    plt.imshow(diff_thresh, cmap="gray")
    plt.subplot(2, 3, 3)
    plt.title("MOG2 Mask")
    plt.imshow(mog2_mask, cmap="gray")
    plt.subplot(2, 3, 4)
    plt.title("KNN Mask")
    plt.imshow(knn_mask, cmap="gray")
    plt.subplot(2, 3, 5)
    plt.title("Motion Detection")
    plt.imshow(cv2.cvtColor(img_motion, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 3, 6)
    plt.title("Stabilized")
    plt.imshow(cv2.cvtColor(img_stabilized, cv2.COLOR_BGR2RGB))
    plt.savefig("video_analysis_output.png")
    print("Visualization: Analysis results saved as video_analysis_output.png")

    # %% [9. Interview Scenario: Video Analysis]
    """
    Interview Scenario: Video Analysis
    Q: How does MOG2 background subtraction work?
    A: MOG2 models each pixel as a mixture of Gaussians to separate foreground from background.
    Key: Robust to lighting changes and dynamic backgrounds.
    Example: cv2.createBackgroundSubtractorMOG2()
    """

# Execute the demo
if __name__ == "__main__":
    run_video_analysis_demo()