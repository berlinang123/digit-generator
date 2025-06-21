# app.py
import tensorflow as tf
import numpy as np
import gradio as gr

# Load model once
model = tf.keras.models.load_model("model/digit_generator_model.keras")

def generate_digits(digit: int):
    noise = tf.random.normal([5, 100])
    labels = tf.constant([digit] * 5, dtype=tf.int32)
    generated = model([noise, labels], training=False)
    generated = (generated + 1) / 2.0  # rescale to [0, 1]
    images = [img.numpy().squeeze() for img in generated]
    return images

iface = gr.Interface(
    fn=generate_digits,
    inputs=gr.Slider(0, 9, step=1, label="Choose a Digit"),
    outputs=[gr.Image(shape=(28, 28), image_mode='L', label=f"Sample {i+1}") for i in range(5)],
    title="Handwritten Digit Generator (0–9)",
    description="Enter a digit, and this app will generate 5 MNIST-style handwritten images using a conditional GAN."
)

if __name__ == "__main__":
    iface.launch()
