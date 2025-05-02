# %% [1. Introduction to Video Basics]
# Learn to read/write videos, extract frames, and access video properties with OpenCV.

# Setup: pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_video_basics_demo():
    # %% [2. Create Synthetic Video Frames]
    # Generate a sequence of 10 frames (300x300) with a moving white square
    frames = []
    for i in range(10):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        x = 100 + i * 10  # Move 10px right per frame
        frame[x:x+50, 100:150] = [255, 255, 255]  # White square
        frames.append(frame)
    print("Synthetic Video: 10 frames with moving white square created")

    # %% [3. Write Video]
    height, width = 300, 300
    fps = 10
    out = cv2.VideoWriter("synthetic_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print("Video Written: synthetic_video.mp4 saved")

    # %% [4. Read Video and Extract Frames]
    cap = cv2.VideoCapture("synthetic_video.mp4")
    extracted_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        extracted_frames.append(frame)
    cap.release()
    print(f"Video Read: {len(extracted_frames)} frames extracted")

    # %% [5. Video Properties]
    cap = cv2.VideoCapture("synthetic_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Properties: FPS={fps}, Resolution={width}x{height}, Frames={frame_count}")
    cap.release()

    # %% [6. Visualization]
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate([0, 4, 9]):  # Show first, middle, last frames
        plt.subplot(1, 3, i+1)
        plt.title(f"Frame {idx+1}")
        plt.imshow(cv2.cvtColor(extracted_frames[idx], cv2.COLOR_BGR2RGB))
    plt.savefig("video_basics_output.png")
    print("Visualization: Frames saved as video_basics_output.png")

    # %% [7. Interview Scenario: Video Basics]
    """
    Interview Scenario: Video Basics
    Q: How do you read a video and extract its properties in OpenCV?
    A: Use cv2.VideoCapture to read frames and get properties like FPS, resolution.
    Key: Properties are accessed via cap.get(cv2.CAP_PROP_*).
    Example: cap.get(cv2.CAP_PROP_FPS)
    """

# Execute the demo
if __name__ == "__main__":
    run_video_basics_demo()