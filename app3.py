import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("image_model.h5")

# Class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat',
               'Deer', 'Dog', 'Frog', 'Horse',
               'Ship', 'Truck']

# Title
st.title("Image Identification App")

st.write("Upload an image and TensorFlow will identify it.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file)

    # Show image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Resize image
    img = img.resize((32, 32))

    # Convert image to array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Get class
    predicted_class = class_names[np.argmax(prediction)]

    # Confidence
    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"Prediction: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")