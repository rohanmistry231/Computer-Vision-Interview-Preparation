# %% [1. Introduction to Optical Flow]
# Learn dense and sparse optical flow and motion tracking with OpenCV.

# Setup: pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_optical_flow_demo():
    # %% [2. Create Synthetic Video Frames]
    # Generate two 300x300 frames with a moving white square
    frame1 = np.zeros((300, 300, 3), dtype=np.uint8)
    frame2 = np.zeros((300, 300, 3), dtype=np.uint8)
    frame1[100:150, 100:150] = [255, 255, 255]  # White square
    frame2[110:160, 110:160] = [255, 255, 255]  # Shifted 10px right, down
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    print("Synthetic Frames: Two frames with moving white square created")

    # %% [3. Dense Optical Flow (Farneback)]
    flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    img_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    print("Dense Optical Flow: Farneback flow computed")

    # %% [4. Sparse Optical Flow (Lucas-Kanade)]
    corners = cv2.goodFeaturesToTrack(frame1_gray, maxCorners=10, qualityLevel=0.3, minDistance=7)
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, corners, None)
    img_lk = frame2.copy()
    for i, (new, old) in enumerate(zip(p1[st==1], corners[st==1])):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(img_lk, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        cv2.circle(img_lk, (int(a), int(b)), 5, (0, 0, 255), -1)
    print("Sparse Optical Flow: Lucas-Kanade tracked keypoints")

    # %% [5. Motion Tracking]
    # Simulate tracking by drawing bounding box around moving square
    x, y = int(p1[st==1][0][0]), int(p1[st==1][0][1])
    img_track = frame2.copy()
    cv2.rectangle(img_track, (x-25, y-25), (x+25, y+25), (0, 0, 255), 2)
    print("Motion Tracking: Bounding box drawn around moving square")

    # %% [6. Visualization]
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    plt.title("Frame 1")
    plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 2, 2)
    plt.title("Frame 2")
    plt.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 2, 3)
    plt.title("Dense Flow")
    plt.imshow(cv2.cvtColor(img_flow, cv2.COLOR_BGR2RGB))
    plt.subplot(2, 2, 4)
    plt.title("Lucas-Kanade")
    plt.imshow(cv2.cvtColor(img_lk, cv2.COLOR_BGR2RGB))
    plt.savefig("optical_flow_output.png")
    print("Visualization: Optical flow results saved as optical_flow_output.png")

    # %% [7. Interview Scenario: Optical Flow]
    """
    Interview Scenario: Optical Flow
    Q: What are the applications of optical flow in computer vision?
    A: Optical flow estimates motion for tracking, video stabilization, and activity recognition.
    Key: Dense flow (Farneback) is global; sparse flow (Lucas-Kanade) tracks specific points.
    Example: cv2.calcOpticalFlowFarneback(frame1, frame2, ...)
    """

# Execute the demo
if __name__ == "__main__":
    run_optical_flow_demo()