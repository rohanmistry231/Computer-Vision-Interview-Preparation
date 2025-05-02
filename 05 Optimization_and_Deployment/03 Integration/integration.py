# %% [1. Introduction to Integration]
# Learn OpenCV integration with TensorFlow, ROS, and Raspberry Pi.

# Setup: pip install opencv-python numpy matplotlib tensorflow
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

def run_integration_demo():
    # %% [2. Create Synthetic Image]
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 150), (200, 200), (255, 255, 255), -1)  # Car-like shape
    print("Synthetic Image: 300x300 with car-like shape created")

    # %% [3. OpenCV with TensorFlow]
    model = MobileNetV2(weights='imagenet')
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_preds = decode_predictions(predictions, top=1)[0]
    img_tf = img.copy()
    label = f"{decoded_preds[0][1]}: {decoded_preds[0][2]:.2f}"
    cv2.putText(img_tf, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(f"OpenCV with TensorFlow: Predicted {decoded_preds[0][1]} ({decoded_preds[0][2]:.2f})")

    # %% [4. OpenCV with ROS (Simulated)]
    """
    ROS Node Example (Conceptual):
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    def image_callback(msg):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        edges = cv2.Canny(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY), 100, 200)
        pub.publish(bridge.cv2_to_imgmsg(edges, "mono8"))
    rospy.init_node('vision_node')
    pub = rospy.Publisher('edges', Image, queue_size=10)
    sub = rospy.Subscriber('camera/image', Image, image_callback)
    rospy.spin()
    """
    img_ros = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
    print("OpenCV with ROS: Simulated ROS node for edge detection")

    # %% [5. OpenCV in Embedded Systems (Raspberry Pi, Conceptual)]
    """
    Raspberry Pi Setup:
    1. Install Raspbian and Python 3.8+.
    2. Install OpenCV: pip install opencv-python.
    3. Connect camera module.
    4. Run lightweight vision pipeline:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200)
        cv2.imshow('Edges', edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    """
    img_pi = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
    print("OpenCV in Raspberry Pi: Simulated lightweight edge detection")

    # %% [6. Visualization]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("TensorFlow Integration")
    plt.imshow(cv2.cvtColor(img_tf, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title("ROS Integration (Simulated)")
    plt.imshow(img_ros, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Raspberry Pi (Simulated)")
    plt.imshow(img_pi, cmap="gray")
    plt.savefig("integration_output.png")
    print("Visualization: Integration results saved as integration_output.png")

    # %% [7. Interview Scenario: Integration]
    """
    Interview Scenario: Integration
    Q: How do you integrate OpenCV with TensorFlow for real-time inference?
    A: Use OpenCV for image preprocessing (e.g., resize, color conversion) and TensorFlow for model inference (e.g., MobileNetV2).
    Key: Ensure compatible data formats (e.g., BGR to RGB, normalized inputs).
    Example: cv2.resize(img, (224, 224)); model.predict(img_array)
    """

# Execute the demo
if __name__ == "__main__":
    run_integration_demo()