# %% [1. Introduction to Deep Learning-Based Detection]
# Learn YOLOv3 object detection with OpenCV and TensorFlow.

# Setup: pip install opencv-python numpy matplotlib tensorflow
# Download: yolov3.weights, yolov3.cfg, coco.names (see README)
import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_deep_learning_detection_demo():
    # %% [2. Create Synthetic Image]
    # Generate a 300x300 image with a car-like shape (rectangle + wheels)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 150), (200, 200), (255, 255, 255), -1)  # Car body
    cv2.circle(img, (120, 200), 10, (0, 0, 0), -1)  # Left wheel
    cv2.circle(img, (180, 200), 10, (0, 0, 0), -1)  # Right wheel
    print("Synthetic Image: 300x300 with car-like shape created")

    # %% [3. Load YOLOv3 Model]
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print("YOLOv3: Model and COCO classes loaded")

    # %% [4. Preprocess and Detect Objects]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    height, width = img.shape[:2]
    boxes = []
    confidences = []
    class_ids = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Class 2 is 'car' in COCO
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    img_yolo = img.copy()
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(img_yolo, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img_yolo, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(f"YOLOv3 Detection: {len(indices)} objects detected (car-like shape)")

    # %% [5. Visualization]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title("YOLOv3 Detection")
    plt.imshow(cv2.cvtColor(img_yolo, cv2.COLOR_BGR2RGB))
    plt.savefig("deep_learning_detection_output.png")
    print("Visualization: YOLOv3 results saved as deep_learning_detection_output.png")

    # %% [6. Interview Scenario: Deep Learning-Based Detection]
    """
    Interview Scenario: Deep Learning-Based Detection
    Q: Compare YOLO, SSD, and Faster R-CNN for object detection.
    A: YOLO is fast, single-pass, ideal for real-time; SSD balances speed and accuracy; Faster R-CNN is accurate but slower due to region proposals.
    Key: YOLOv3 uses a single network with grid-based predictions.
    Example: cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    """

# Execute the demo
if __name__ == "__main__":
    run_deep_learning_detection_demo()