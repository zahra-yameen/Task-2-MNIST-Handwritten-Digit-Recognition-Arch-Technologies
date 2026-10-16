
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Set random seeds for more reproducible results
np.random.seed(42)
tf.random.set_seed(42)


# Get the main project folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATA_PATH = os.path.join(BASE_DIR, "dataset", "train.csv")

# Model save directory and path
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_model.keras")


# Create the models folder if it does not exist
os.makedirs(MODEL_DIR, exist_ok=True)

# 3. Load the Kaggle dataset
print("Loading MNIST dataset...")

data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)
print("First five rows:")
print(data.head())

# 4. Separate the correct digit labels from pixel values
if "label" not in data.columns:
    raise ValueError(
        "The CSV must contain a 'label' column. "
        "Please check that you downloaded train.csv."
    )

y = data["label"].astype("int32").to_numpy()
X = data.drop(columns=["label"]).to_numpy(dtype="float32")

# Check that each image has exactly 784 pixels
if X.shape[1] != 784:
    raise ValueError(
        f"Expected 784 pixel columns, but found {X.shape[1]}."
    )

# 5. Preprocess the pixel values
# Original pixel values range from 0 to 255.
# Divide by 255 to convert them to values between 0 and 1.
X = X / 255.0

# 6. Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining images:", X_train.shape[0])
print("Validation images:", X_val.shape[0])

# 7. Build the neural network
model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# 8. Configure the model for training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display the model structure
model.summary()

# 9. Train the model
print("\nStarting model training...")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_val, y_val),
    verbose=1
)

# 10. Evaluate the trained model
print("\nEvaluating the model...")

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print(f"\nValidation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

# 11. Predict digits for validation images
probabilities = model.predict(X_val, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy score: {accuracy * 100:.2f}%")

# 12. Display precision, recall and F1-score
print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        labels=list(range(10)),
        zero_division=0
    )
)

# 13. Display the confusion matrix
cm = confusion_matrix(y_val, y_pred, labels=list(range(10)))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(range(10))
)

display.plot(cmap="Blues", values_format="d")
plt.title("MNIST Digit Recognition - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 14. Plot training and validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_graph.png")
plt.show()

# 15. Save the trained model
model.save(MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")

# 16. Show some sample predictions
plt.figure(figsize=(12, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    # Reconstruct a 28x28 image from its 784 pixels
    image = X_val[i].reshape(28, 28)

    plt.imshow(image, cmap="gray")
    plt.title(
        f"Actual: {y_val[i]}\nPredicted: {y_pred[i]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY!")
