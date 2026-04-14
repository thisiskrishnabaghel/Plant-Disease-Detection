import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("model.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

def predict(model, img):
    img = img.convert("RGB")
    img = img.resize((255, 255))

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0])) * 100

    return predicted_class, confidence

# UI
st.title("🌿 Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image")

if uploaded_file:
    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    pred, conf = predict(model, img)

    st.write("### Prediction:", pred)
    st.write("### Confidence:", f"{conf:.2f}%")
