import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import torch.nn.functional as F
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and processor
processor = AutoImageProcessor.from_pretrained("Molkaatb/Liveness_Vit")
model = AutoModelForImageClassification.from_pretrained("Molkaatb/Liveness_Vit")

# Custom CSS for better styling
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
        color: #333;
    }
    .small-image {
        width: 300px;
        margin: auto;
        display: block;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    .real {
        color: green;
    }
    .spoof {
        color: red;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Real or Spoof Image Classification")

# Upload image option
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.image = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img_mirrored = cv2.flip(img_rgb, 1)  # Mirror the image horizontally
        self.image = img_mirrored
        return av.VideoFrame.from_ndarray(img_mirrored, format="rgb24")

ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

def classify_image(image):
    # Convert the captured image to PIL Image
    image = Image.fromarray(np.uint8(image)).convert('RGB')

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Calculate probabilities
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_idx = logits.argmax(-1).item()

    # Map the predicted index to a label
    labels = model.config.id2label
    predicted_label = labels[predicted_class_idx]

    return predicted_label, probabilities

def display_result(image, predicted_label, probabilities):
    st.image(image, caption='Image', use_column_width=False, width=300)
    st.write("")

    # Display prediction
    label_class = "real" if predicted_label.lower() == "real" else "spoof"
    st.markdown(f'<p class="result {label_class}">Prediction: {predicted_label}</p>', unsafe_allow_html=True)

    # Display probabilities for each class using a styled bar chart
    labels = model.config.id2label.values()
    probabilities = probabilities[0].numpy()

    fig, ax = plt.subplots(figsize=(4, 2.5))
    sns.barplot(x=probabilities, y=list(labels), palette=['#8bc34a' if label.lower() == 'real' else '#ff5722' for label in labels], ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability', fontsize=10)
    ax.set_title('Class Probabilities', fontsize=12)
    sns.despine(left=True, bottom=True)

    for i, (value, name) in enumerate(zip(probabilities, labels)):
        ax.text(value, i, f'{value:.2f}', color='black', ha="left", va="center", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)

if ctx.video_transformer:
    image = ctx.video_transformer.image
    if image is not None:
        st.write("Classifying...")
        predicted_label, probabilities = classify_image(image)
        display_result(image, predicted_label, probabilities)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("Classifying...")
    predicted_label, probabilities = classify_image(image)
    display_result(np.array(image), predicted_label, probabilities)
