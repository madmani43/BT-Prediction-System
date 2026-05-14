import os
import numpy as np
import gradio as gr
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']


def predict(image):
    if image is None:
        return "No image provided"

    try:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]

        image = tf.image.resize(image, [128, 128]).numpy()
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        label = class_labels[predicted_index] if predicted_index < len(class_labels) else str(predicted_index)

        if label == 'notumor':
            return f"No Tumor Detected ({confidence * 100:.2f}%)"
        return f"Tumor Detected: {label} ({confidence * 100:.2f}%)"

    except Exception as e:
        return f"Prediction error: {e}"


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload MRI Image"),
    outputs="text",
    title="MRI Brain Tumor Detection",
    description="Upload a brain MRI image and the model will predict whether a tumor is present and its type."
).launch()