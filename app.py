from pathlib import Path

import streamlit as st
import tensorflow as tf

# ------------------------------------------------------------------ config
MODEL_PATH = Path(__file__).parent / "models" / "pneumonia_mobilenetv2.keras"
IMG_SIZE = (224, 224)                 # same as training
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # 0 = NORMAL, 1 = PNEUMONIA
THRESHOLD = 0.5

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="centered")


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image_bytes: bytes):
    """Same steps as load_image() in the notebook: grayscale -> 224x224 -> float32.
    No scaling here, the model has a Rescaling layer inside."""
    image = tf.image.decode_image(image_bytes, channels=1, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, axis=0)   # (1, 224, 224, 1)


def predict(model, image_bytes: bytes):
    x = preprocess(image_bytes)
    p_pneumonia = float(model(x, training=False).numpy()[0][0])
    index = int(p_pneumonia >= THRESHOLD)
    confidence = p_pneumonia if index == 1 else 1 - p_pneumonia
    return CLASS_NAMES[index], confidence, p_pneumonia


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "CNN classifier (MobileNetV2 transfer learning, 224×224 grayscale) "
        "trained to separate **NORMAL** and **PNEUMONIA** chest X-rays."
    )
    st.caption(
        "Reported on the test set: sensitivity 96.7%, specificity 80.3%, "
        "accuracy 90.5%. The test set was also used for early stopping, "
        "so real-world performance may be lower."
    )
    st.warning("Works only for frontal chest X-rays. Other images will still get a prediction, but it is meaningless.")

# ------------------------------------------------------------------ main page
st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to get a prediction.")
st.info("Educational project, not a medical device. Do not use it for diagnosis.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH.name}`. Put it in the same folder as `app.py`.")
    st.stop()

model = load_model()

uploaded = st.file_uploader("Chest X-ray image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    image_bytes = uploaded.getvalue()

    try:
        label, confidence, p_pneumonia = predict(model, image_bytes)
    except Exception:
        st.error("Could not read this file as an image. Try a JPG or PNG.")
        st.stop()

    col_img, col_result = st.columns(2)

    with col_img:
        st.image(image_bytes, caption=uploaded.name, use_container_width=True)

    with col_result:
        if label == "PNEUMONIA":
            st.error("### Prediction: PNEUMONIA")
        else:
            st.success("### Prediction: NORMAL")

        st.metric("Confidence", f"{confidence * 100:.2f}%")

        st.write("**Class probabilities**")
        st.progress(p_pneumonia, text=f"PNEUMONIA: {p_pneumonia * 100:.2f}%")
        st.progress(1 - p_pneumonia, text=f"NORMAL: {(1 - p_pneumonia) * 100:.2f}%")
