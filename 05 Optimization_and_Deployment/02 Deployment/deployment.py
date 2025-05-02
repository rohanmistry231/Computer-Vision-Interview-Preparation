# %% [1. Introduction to Deployment]
# Learn Flask for web apps, Docker containers, and real-time inference with OpenCV.

# Setup: pip install opencv-python numpy matplotlib tensorflow flask
# Optional: Install Docker for containerization
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import base64
from io import BytesIO

app = Flask(__name__)

def run_deployment_demo():
    # %% [2. Load Pretrained Model]
    model = MobileNetV2(weights='imagenet')
    print("Deployment: Loaded MobileNetV2 for real-time inference")

    # %% [3. Create Synthetic Image]
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 150), (200, 200), (255, 255, 255), -1)  # Car-like shape
    print("Synthetic Image: 300x300 with car-like shape created")

    # %% [4. Flask App for Real-Time Inference]
    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        img_data = base64.b64decode(data['image'])
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_array)
        predictions = model.predict(img_array)
        decoded_preds = decode_predictions(predictions, top=1)[0]
        return jsonify({'prediction': decoded_preds[0][1], 'confidence': float(decoded_preds[0][2])})
    
    # Simulate Flask inference with synthetic image
    _, img_encoded = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    simulated_request = {'image': img_base64}
    with app.test_request_context('/predict', json=simulated_request, method='POST'):
        response = predict()
    print(f"Flask Inference: Simulated prediction: {response.json['prediction']} ({response.json['confidence']:.2f})")

    # %% [5. Docker Container (Conceptual)]
    """
    Dockerfile Example:
    FROM python:3.8-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["flask", "run", "--host=0.0.0.0"]
    
    Build and Run:
    docker build -t vision-app .
    docker run -p 5000:5000 vision-app
    """
    print("Docker: Conceptual Dockerfile provided for containerizing Flask app")

    # %% [6. Real-Time Inference]
    start_time = time.time()
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_array)
    predictions = model.predict(img_array)
    inference_time = time.time() - start_time
    decoded_preds = decode_predictions(predictions, top=1)[0]
    img_pred = img.copy()
    label = f"{decoded_preds[0][1]}: {decoded_preds[0][2]:.2f}"
    cv2.putText(img_pred, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(f"Real-Time Inference: Predicted {decoded_preds[0][1]} in {inference_time:.2f} seconds")

    # %% [7. Visualization]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title("Inference Result")
    plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    plt.savefig("deployment_output.png")
    print("Visualization: Deployment results saved as deployment_output.png")

    # %% [8. Interview Scenario: Deployment]
    """
    Interview Scenario: Deployment
    Q: What are the benefits of using Docker for deploying vision applications?
    A: Docker ensures consistent environments, simplifies dependency management, and enables scalability.
    Key: Containers isolate the app, making it portable across platforms (e.g., AWS, GCP).
    Example: docker build -t vision-app .
    """

# Execute the demo
if __name__ == "__main__":
    run_deployment_demo()