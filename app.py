# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/digit_generator_model.keras")

generator = load_model()

st.set_page_config(page_title="Digit Generator", layout="centered")
st.title("🖋️ Handwritten Digit Generator")
st.markdown("Enter a digit (0 to 9) and generate 5 handwritten images using the trained model.")

digit = st.number_input("Choose a digit", min_value=0, max_value=9, step=1)

if st.button("Generate Images"):
    st.success(f"Generated 5 images for digit: {digit}")

    noise = tf.random.normal([5, 100])
    labels = tf.constant([digit] * 5, dtype=tf.int32)
    gen_images = generator([noise, labels], training=False)
    gen_images = (gen_images + 1) / 2.0  # Scale to [0,1]

    # Display images
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(gen_images[i, :, :, 0].numpy(), width=100, clamp=True, channels="L")