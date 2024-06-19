import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from tensorflow.keras.models import load_model
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings

# Load the trained model
model = load_model('D:\Download\Img_Dog_classifier\MNIST\mnist_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = image.convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model
    return img_array

# Streamlit app
st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="centered")

# Title and description
st.title('🖌️ Handwritten Digit Recognition')
st.markdown("""
    This app uses a neural network model trained on the MNIST dataset to predict handwritten digits.
    Capture an image of a handwritten digit, and the model will predict what digit it is.
""")

# Sidebar with instructions
st.sidebar.header('Instructions')
st.sidebar.markdown("""
    1. Capture a clear image of a handwritten digit.
    2. The image should be centered and have minimal noise.
    3. The app will preprocess the image and display the prediction.
""")

# Define a video transformer to process the video frames
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.predicted_digit = None
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # Enhance image quality (example: contrast enhancement)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.5)  # Increase contrast

        # Preprocess the image
        preprocessed_image = preprocess_image(pil_img)

        # Predict the digit
        prediction = model.predict(preprocessed_image)
        self.predicted_digit = np.argmax(prediction, axis=1)[0]

        # Draw the prediction on the frame
        cv2.putText(img, f'Predicted Digit: {self.predicted_digit}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.frame = img
        return img

# Streamlit-WeRTC client settings
webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoTransformer,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    ),
)

# Button to capture the current frame
if webrtc_ctx.video_processor:
    if st.button("Capture Image"):
        captured_frame = webrtc_ctx.video_processor.frame
        if captured_frame is not None:
            st.image(captured_frame, caption='Captured Image', use_column_width=True)
            st.write(f"Predicted Digit: {webrtc_ctx.video_processor.predicted_digit}")

# Add a footer with GitHub link or any additional info
st.markdown("""
    <hr style="border:2px solid gray"> </hr>
    <p style='text-align: center;'>
    Developed by <a href="https://github.com/your-github-username" target="_blank">Your Name</a> | 
    <a href="https://github.com/your-github-username/your-repo-name" target="_blank">GitHub Repo</a>
    </p>
    """, unsafe_allow_html=True)
