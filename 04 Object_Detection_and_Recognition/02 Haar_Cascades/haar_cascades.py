# %% [1. Introduction to Haar Cascades]
# Learn face and eye detection using pretrained Haar cascades with OpenCV.

# Setup: pip install opencv-python numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_haar_cascades_demo():
    # %% [2. Create Synthetic Face Image]
    # Generate a 300x300 image with a simplified face (ovals for face and eyes)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.ellipse(img, (150, 150), (100, 120), 0, 0, 360, (200, 200, 200), -1)  # Face
    cv2.ellipse(img, (120, 120), (20, 30), 0, 0, 360, (50, 50, 50), -1)  # Left eye
    cv2.ellipse(img, (180, 120), (20, 30), 0, 0, 360, (50, 50, 50), -1)  # Right eye
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Synthetic Image: 300x300 with simplified face and eyes created")

    # %% [3. Load Haar Cascade Classifiers]
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    print("Haar Cascades: Loaded face and eye classifiers")

    # %% [4. Face Detection]
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    img_faces = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img_faces, (x, y), (x+w, y+h), (0, 0, 255), 2)
    print(f"Face Detection: {len(faces)} faces detected")

    # %% [5. Eye Detection]
    eyes = eye_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    img_eyes = img.copy()
    for (x, y, w, h) in eyes:
        cv2.rectangle(img_eyes, (x, y), (x+w, y+h), (0, 255, 0), 2)
    print(f"Eye Detection: {len(eyes)} eyes detected")

    # %% [6. Visualization]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title("Face Detection")
    plt.imshow(cv2.cvtColor(img_faces, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 3)
    plt.title("Eye Detection")
    plt.imshow(cv2.cvtColor(img_eyes, cv2.COLOR_BGR2RGB))
    plt.savefig("haar_cascades_output.png")
    print("Visualization: Haar cascade results saved as haar_cascades_output.png")

    # %% [7. Interview Scenario: Haar Cascades]
    """
    Interview Scenario: Haar Cascades
    Q: What are the limitations of Haar cascades for face detection?
    A: Haar cascades are fast but struggle with non-frontal faces, occlusions, and varying lighting.
    Key: Use integral images for efficiency but less robust than deep learning.
    Example: cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    """

# Execute the demo
if __name__ == "__main__":
    run_haar_cascades_demo()