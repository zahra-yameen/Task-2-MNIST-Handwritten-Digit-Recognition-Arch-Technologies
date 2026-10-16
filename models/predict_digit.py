
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# Find the main project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get the full path of the saved model
model_path = os.path.join(BASE_DIR, "models", "mnist_model.keras")

# Check whether the model file exists
print("Model path:", model_path)
print("Model file exists:", os.path.isfile(model_path))

# Load the trained model
model = tf.keras.models.load_model(model_path)

print("MNIST model loaded successfully!")


# Load the same labeled dataset
DATA_PATH = os.path.join(BASE_DIR, "dataset", "train.csv")
data = pd.read_csv(DATA_PATH)

# Select one image to test
image_number = 0

# Get its pixels and correct label
pixels = data.drop(columns=["label"]).iloc[image_number].to_numpy(
    dtype=np.float32
)
actual_digit = int(data["label"].iloc[image_number])

# Preprocess the image exactly as during training
pixels = pixels / 255.0

# Make a prediction
probabilities = model.predict(
    pixels.reshape(1, 784),
    verbose=0
)

predicted_digit = int(np.argmax(probabilities[0]))
confidence = float(np.max(probabilities[0])) * 100

# Display the result
print("Actual digit:", actual_digit)
print("Predicted digit:", predicted_digit)
print(f"Model confidence score: {confidence:.2f}%")

plt.imshow(pixels.reshape(28, 28), cmap="gray")
plt.title(
    f"Actual: {actual_digit} | Predicted: {predicted_digit}"
)
plt.axis("off")
plt.show()
