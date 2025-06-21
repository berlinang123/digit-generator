# model.py
import numpy as np

def generate_digit_images(digit, num_samples=5):
    # TODO: Replace with your trained model
    images = []
    for _ in range(num_samples):
        img = np.random.normal(loc=int(digit) * 10, scale=40, size=(28, 28))
        img = np.clip(img, 0, 255).astype(np.uint8)
        images.append(img)
    return images
