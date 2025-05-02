# %% [1. Introduction to Image Classification]
# Learn pretrained MobileNetV2 classification and transfer learning with TensorFlow.

# Setup: pip install opencv-python numpy matplotlib tensorflow
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

def run_image_classification_demo():
    # %% [2. Load CIFAR-10 Dataset]
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    img = x_train[0]  # Example: airplane image
    print("CIFAR-10: Loaded sample image for classification")

    # %% [3. Pretrained MobileNetV2 Classification]
    model = MobileNetV2(weights='imagenet')
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_array)
    predictions = model.predict(img_preprocessed)
    decoded_preds = decode_predictions(predictions, top=3)[0]
    img_pred = img.copy()
    label = f"{decoded_preds[0][1]}: {decoded_preds[0][2]:.2f}"
    cv2.putText(img_pred, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(f"MobileNetV2: Top prediction: {decoded_preds[0][1]} ({decoded_preds[0][2]:.2f})")

    # %% [4. Transfer Learning (Simulated)]
    # Simulate fine-tuning MobileNetV2 for CIFAR-10 (subset)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train[:1000], y_train[:1000], epochs=1, batch_size=32, verbose=0)
    pred = model.predict(x_train[:1])
    class_id = np.argmax(pred[0])
    print(f"Transfer Learning: Predicted class for sample: {class_names[class_id]}")

    # %% [5. Visualization]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title("MobileNetV2 Prediction")
    plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    plt.savefig("image_classification_output.png")
    print("Visualization: Classification results saved as image_classification_output.png")

    # %% [6. Interview Scenario: Image Classification]
    """
    Interview Scenario: Image Classification
    Q: How does transfer learning improve image classification?
    A: Transfer learning uses pretrained models (e.g., MobileNetV2) to leverage learned features, reducing training time and data needs.
    Key: Fine-tune top layers for specific tasks like CIFAR-10 classification.
    Example: MobileNetV2(weights='imagenet', include_top=False)
    """

# Execute the demo
if __name__ == "__main__":
    run_image_classification_demo()