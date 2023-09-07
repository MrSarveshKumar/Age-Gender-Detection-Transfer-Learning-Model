import streamlit as st
import pickle
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


gender_mapping = {
    1: 'Female',
    0: 'Male'
}


def get_image_features(image):
  img = Image.open(image)
  img = img.convert('L')
  img = img.resize((128, 128), Image.ANTIALIAS)
  img = np.array(img)
  img = img.reshape(1, 128, 128, 1)
  img = img / 255.0
  return img



st.title("Age-Gender Detection App")
st.write("Upload an image to predict the age and gender.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if st.button("Predict") and uploaded_file is not None:
    
    img_to_test = uploaded_file
    features = get_image_features(img_to_test)
    pred = loaded_model.predict(features)
    gender = gender_mapping[round(pred[0][0][0])]
    age = round(pred[1][0][0])

    st.subheader("Prediction Result")
    st.write("")
    st.success(f'Predicted Age: {age}  \n\nPredicted Gender: {gender}')
    plt.axis('off')
    plt.imshow(np.array(Image.open(img_to_test)))
    st.pyplot()


else:
    st.write("Please upload an image before predicting.")


st.markdown("****")

st.write("NOTE: This is only for Educational Purpose")
st.write("<span style='font-size: 15px;'>Founder: *Sarvesh Kumar*</span>", unsafe_allow_html=True)