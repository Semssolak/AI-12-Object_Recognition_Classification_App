import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions

# Load the InceptionResNetV2 model with pretrained weights
model = InceptionResNetV2(weights='imagenet')

# List of recognizable categories from the ImageNet dataset (shortened for demo purposes)
categories = """
Tench, Goldfish, White Shark, Tiger Shark, Hammerhead, Electric Ray, Stingray, Cock, Hen, Ostrich,
Pelican, Penguin, Parrot, Goose, Bald Eagle, Koala, Chimpanzee, Orangutan, Tiger, Lion, Leopard,
Elephant, Zebra, Camel, Kangaroo, Submarine, Fire Engine, Taxi, Race Car, Bicycle, Motorcycle,
Sports Car, Airplane, Warplane, Helicopter
"""

# Streamlit app
st.title("Image Classification")
st.write(
    "Upload an image, and the model will classify it using InceptionResNetV2. Please only upload images related to the recognizable categories to ensure accurate predictions.")

# Button to show recognizable categories
if st.button("Show Recognizable Categories"):
    st.write(categories)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open and preprocess the image
    img = Image.open(uploaded_file).resize((299, 299))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = np.array(img)
    img = img.reshape(-1, 299, 299, 3)
    img = preprocess_input(img)

    # Make prediction
    preds = model.predict(img)

    # Decode predictions
    decoded_preds = decode_predictions(preds, top=1)[0][0]

    # Display the result
    st.write(f"Predicted: {decoded_preds[1]} with a confidence of {decoded_preds[2] * 100:.2f}%")


#streamlit run main.py
