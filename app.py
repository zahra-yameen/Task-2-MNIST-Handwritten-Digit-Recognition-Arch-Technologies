
import os

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("MNIST Handwritten Digit Recognition")

st.write(
    "Upload an image of a handwritten digit from 0 to 9. "
    "The trained deep learning model will predict the digit."
)

st.info(
    "For best results, upload one clearly written digit "
    "on a plain, light background."
)

# --------------------------------------------------
# 2. LOAD THE TRAINED KERAS MODEL
# --------------------------------------------------

MODEL_PATH = os.path.join("models", "mnist_model.keras")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Please check that "
        "models/mnist_model.keras exists."
    )
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(
        "The Keras model could not be loaded. "
        "Check your TensorFlow installation and model file."
    )
    st.exception(error)
    st.stop()

st.success("Trained Keras model loaded successfully!")

# --------------------------------------------------
# 3. UPLOAD AN IMAGE
# --------------------------------------------------

st.subheader("Upload a handwritten digit")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

# --------------------------------------------------
# 4. PREPROCESS THE IMAGE
# --------------------------------------------------

if uploaded_file is not None:
    try:
        # Convert the image to grayscale
        image = Image.open(uploaded_file).convert("L")

        st.subheader("Original Image")
        st.image(
            image,
            caption="Uploaded image",
            width=250
        )

        # Improve contrast
        image = ImageOps.autocontrast(image)

        # Resize to the model's expected image dimensions
        image = ImageOps.pad(
            image,
            (28, 28),
            color=255
        )

        # MNIST commonly uses a black background
        # and a bright handwritten digit.
        if np.asarray(image).mean() > 127:
            image = ImageOps.invert(image)

        # Normalize pixels to the range 0 to 1
        pixels = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        # Add the channel dimension:
        # (28, 28) -> (28, 28, 1)
        # Then add the batch dimension:
        # (28, 28, 1) -> (1, 28, 28, 1)
        input_image = pixels.reshape(1, 28, 28, 1)

        # --------------------------------------------------
        # 5. MAKE A PREDICTION
        # --------------------------------------------------

        predictions = model.predict(
            input_image,
            verbose=0
        )

        # Find the digit with the highest output score
        predicted_digit = int(np.argmax(predictions[0]))

        st.subheader("Processed Image")
        st.image(
            image,
            caption="Preprocessed 28 × 28 image",
            width=150
        )

        st.subheader("Prediction Result")

        st.metric(
            label="Recognized Digit",
            value=str(predicted_digit)
        )

        st.success(
            f"The model predicts digit {predicted_digit}."
        )

        # Show model output scores
        st.subheader("Model Output Scores")

        scores = predictions[0]

        score_data = {
            "Digit": list(range(10)),
            "Model output": [float(score) for score in scores]
        }

        st.bar_chart(
            data=score_data,
            x="Digit",
            y="Model output"
        )

        st.caption(
            "Output scores are not necessarily calibrated probabilities. "
            "Prediction quality depends on image formatting and handwriting."
        )

    except Exception as error:
        st.error("Could not process the uploaded image.")
        st.exception(error)

# --------------------------------------------------
# 6. PROJECT INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("About This Project")

st.write(
    """
    This project uses a trained TensorFlow/Keras model to recognize
    handwritten digits from 0 to 9.

    The model was trained using labeled digit images from the
    Kaggle MNIST Digit Recognizer dataset.

    Technologies:
    - Python
    - TensorFlow / Keras
    - NumPy
    - Pillow
    - Streamlit
    """
)

st.caption("MNIST Digit Recognition — Machine Learning Assignment")
