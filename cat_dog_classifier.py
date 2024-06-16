import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import requests
from io import BytesIO
from PIL import UnidentifiedImageError
import logging
import os

# Page configuration (must be the first Streamlit command)
st.set_page_config(page_title="🐾 Cat vs Dog Classifier 🐶", page_icon=":cat:", layout="centered")

# Suppress specific warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses TensorFlow logging (INFO and WARNING messages)

# Suppress absl logging warnings
logging.getLogger('absl').setLevel(logging.ERROR)

# Function to load the model with caching
@st.cache_resource
def load_model_once(model_path):
    return load_model(model_path)

# Define the path to the model
best_model_path = r'D:\Download\Img_Dog_classifier\best_model (1).h5'

# Load the model and store it in cache
try:
    best_model = load_model_once(best_model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

def load_and_process_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        if 'image' in response.headers['Content-Type']:
            image_data = BytesIO(response.content)
            try:
                pil_image = load_img(image_data, target_size=(224, 224))
                image = img_to_array(pil_image)
                image = image / 255.0
                return image
            except UnidentifiedImageError:
                st.error("Error: Cannot identify the image file.")
                return None
        else:
            st.error("Error: URL does not point to an image.")
            return None
    else:
        st.error(f"Error: Unable to fetch image from URL. Status code: {response.status_code}")
        return None

# Function to classify prediction
def classify(prediction):
    if prediction == 0:
        return "Cat"
    elif prediction == 1:
        return "Dog"
    else:
        return "Unknown"

# Streamlit app
st.title("🐾 Image Classification with Pre-trained Model 🐾")
st.markdown("Upload an image URL to classify it as either a **cat** or a **dog** using a pre-trained model.")
st.markdown("##")

# URL input
st.sidebar.header("Upload Image URL")
image_url = st.sidebar.text_input("Enter the image URL:")

if image_url:
    st.markdown("### Uploaded Image:")
    image = load_and_process_image(image_url)
    if image is not None:
        # Display the image
        st.image(image, caption='Input Image', use_column_width=True, channels="RGB")

        # Expand dimensions to match the model's input shape
        image = np.expand_dims(image, axis=0)

        # Prediction
        with st.spinner('Predicting...'):
            pred = best_model.predict(image)
            CLASS_ID = (pred > 0.5).astype(int)[0][0]

        # Classify and display the result
        classification = classify(CLASS_ID)
        st.success(f"### Prediction: **{classification}** 🐱🐶")

# Additional sidebar information
st.sidebar.markdown("### About")
st.sidebar.info("This app uses a pre-trained neural network to classify images as either a cat or a dog. "
                "Simply enter the URL of the image you want to classify and the model will do the rest.")

# Footer
st.markdown("---")
st.markdown("Developed by [Suraj Vishwakarma](https://www.linkedin.com/in/surajvishwakarma11/)")

