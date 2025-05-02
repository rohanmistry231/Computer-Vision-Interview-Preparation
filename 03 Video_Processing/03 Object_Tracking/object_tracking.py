# %% [1. Introduction to Object Tracking]
# Learn MeanShift, CamShift, KCF, and CSRT trackers with OpenCV.

# Setup: pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_object_tracking_demo():
    # %% [2. Create Synthetic Video Frames]
    # Generate 10 frames (300x300) with a moving white square
    frames = []
    for i in range(10):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        x = 100 + i * 10  # Move 10px right per frame
        frame[x:x+50, 100:150] = [255, 255, 255]  # White square
        frames.append(frame)
    print("Synthetic Video: 10 frames with moving white square created")

    # %% [3. Initialize ROI]
    roi = (100, 100, 50, 50)  # (x, y, w, h) for first frame
    x, y, w, h = roi
    frame_roi = frames[0][y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # %% [4. MeanShift Tracking]
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    mean_shift_results = []
    track_window = roi
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x, y, w, h = track_window
        img_ms = frame.copy()
        cv2.rectangle(img_ms, (x, y), (x+w, y+h), (0, 0, 255), 2)
        mean_shift_results.append(img_ms)
    print("MeanShift Tracking: Tracked square across frames")

    # %% [5. CamShift Tracking]
    cam_shift_results = []
    track_window = roi
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img_cs = frame.copy()
        cv2.polylines(img_cs, [pts], True, (0, 255, 0), 2)
        cam_shift_results.append(img_cs)
    print("CamShift Tracking: Tracked square with adaptive size")

    # %% [6. KCF Tracking]
    kcf_tracker = cv2.TrackerKCF_create()
    kcf_results = []
    kcf_tracker.init(frames[0], roi)
    for frame in frames:
        ok, bbox = kcf_tracker.update(frame)
        img_kcf = frame.copy()
        if ok:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(img_kcf, (x, y), (x+w, y+h), (255, 0, 0), 2)
        kcf_results.append(img_kcf)
    print("KCF Tracking: Tracked square with KCF")

    # %% [7. CSRT Tracking]
    csrt_tracker = cv2.TrackerCSRT_create()
    csrt_results = []
    csrt_tracker.init(frames[0], roi)
    for frame in frames:
        ok, bbox = csrt_tracker.update(frame)
        img_csrt = frame.copy()
        if ok:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(img_csrt, (x, y), (x+w, y+h), (0, 255, 255), 2)
        csrt_results.append(img_csrt)
    print("CSRT Tracking: Tracked square with CSRT")

    # %% [8. Visualization]
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate([0, 4, 9]):  # Show frames 1, 5, 10
        plt.subplot(4, 3, i+1)
        plt.title(f"MeanShift Frame {idx+1}")
        plt.imshow(cv2.cvtColor(mean_shift_results[idx], cv2.COLOR_BGR2RGB))
        plt.subplot(4, 3, i+4)
        plt.title(f"CamShift Frame {idx+1}")
        plt.imshow(cv2.cvtColor(cam_shift_results[idx], cv2.COLOR_BGR2RGB))
        plt.subplot(4, 3, i+7)
        plt.title(f"KCF Frame {idx+1}")
        plt.imshow(cv2.cvtColor(kcf_results[idx], cv2.COLOR_BGR2RGB))
        plt.subplot(4, 3, i+10)
        plt.title(f"CSRT Frame {idx+1}")
        plt.imshow(cv2.cvtColor(csrt_results[idx], cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.savefig("object_tracking_output.png")
    print("Visualization: Tracking results saved as object_tracking_output.png")

    # %% [9. Interview Scenario: Object Tracking]
    """
    Interview Scenario: Object Tracking
    Q: What’s the difference between MeanShift and CSRT trackers?
    A: MeanShift tracks based on color histograms, fast but limited to fixed-size windows; CSRT uses discriminative correlation filters, more robust to scale and rotation.
    Key: CSRT is slower but more accurate for complex scenes.
    Example: cv2.TrackerCSRT_create()
    """

# Execute the demo
if __name__ == "__main__":
    run_object_tracking_demo()