
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Find the model relative to this app.py file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_mobilenetv2.keras"

CLASS_NAMES = [
    "Alternaria Leaf Spot",
    "Bacterial Blight",
    "Fusarium Wilt",
    "Healthy Leaf",
    "Verticillium Wilt",
]

DISEASE_INFO = {
    "Alternaria Leaf Spot":
        "A fungal disease that may cause dark brown leaf spots.",
    "Bacterial Blight":
        "A bacterial disease that may cause dark or angular leaf lesions.",
    "Fusarium Wilt":
        "A soil-associated fungal disease that can cause yellowing and wilting.",
    "Healthy Leaf":
        "The model classified this image as a healthy cotton leaf.",
    "Verticillium Wilt":
        "A soil-associated fungal disease that may cause yellowing and wilting.",
}


@st.cache_resource
def load_disease_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Check the models folder and filename."
        )

    return load_model(
        str(MODEL_PATH),
        custom_objects={"preprocess_input": preprocess_input},
        compile=False,
    )


st.set_page_config(
    page_title="Cotton Disease AI",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 Cotton Disease Detection")
st.write(
    "Upload a cotton leaf image to classify it into one "
    "of five categories using your trained MobileNetV2 model."
)

uploaded_file = st.file_uploader(
    "Choose a cotton leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded cotton leaf", use_container_width=True)

        if st.button("Predict Disease", type="primary"):
            with st.spinner("Analyzing image..."):
                model = load_disease_model()

                image = image.resize(
                    (224, 224),
                    Image.Resampling.BILINEAR,
                )

                # The saved model includes preprocessing internally.
                # Pass RGB pixel values in the 0–255 range.
                image_array = np.asarray(image, dtype=np.float32)
                image_batch = np.expand_dims(image_array, axis=0)

                scores = model.predict(image_batch, verbose=0)[0]

                if len(scores) != len(CLASS_NAMES):
                    raise ValueError(
                        "Model output count does not match the class names."
                    )

                predicted_index = int(np.argmax(scores))
                disease = CLASS_NAMES[predicted_index]
                confidence = float(scores[predicted_index])

                st.success(f"Prediction: {disease}")
                st.metric("Model confidence", f"{confidence:.1%}")
                st.write(DISEASE_INFO[disease])

                st.subheader("Scores by class")
                st.bar_chart({
                    "Score (%)": {
                        name: float(score * 100)
                        for name, score in zip(CLASS_NAMES, scores)
                    }
                })

                st.warning(
                    "This is an experimental image classifier. "
                    "Predictions may be wrong, especially for field "
                    "images that differ from the training dataset. "
                    "Confirm suspected diseases with an agricultural expert."
                )

    except Exception as error:
        st.error("The image could not be analyzed.")
        st.exception(error)
