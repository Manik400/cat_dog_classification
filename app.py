import streamlit as st
import tensorflow as tf
# from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import keras as ks
from keras.api.models import load_model


# Load the model
model = load_model('cd_model.h5')


# Set up the Streamlit app title
# st.title("Dog vs Cat Classifier")

st.title("Dog vs Cat Classifier")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)
    
    # Display the image
    # st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    image = image.resize((256, 256))  # Resize the image to match the input shape
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions
    prediction = model.predict(image)
    
    # Display the prediction
    if prediction[0][0] > 0.5:
        st.write("It's a **Dog**!")
    else:
        st.write("It's a **Cat**!")

