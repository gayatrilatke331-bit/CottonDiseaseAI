
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Cotton Disease AI",
    page_icon="🌿",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_mobilenetv2.keras"

CLASS_NAMES = [
    "Alternaria Leaf Spot",
    "Bacterial Blight",
    "Fusarium Wilt",
    "Healthy Leaf",
    "Verticillium Wilt",
]


# =====================================================
# 2. DISEASE INFORMATION AND MANAGEMENT
# =====================================================

DISEASE_INFO = {
    "Alternaria Leaf Spot": {
        "description": (
            "A fungal leaf disease that can cause brown or "
            "irregular leaf spots, sometimes with concentric rings."
        ),
        "symptoms": [
            "Brown or pale-brown spots on leaves.",
            "Some lesions may develop concentric rings.",
            "Spots may merge into larger blighted areas.",
            "Severely affected leaves may become brittle and fall.",
        ],
        "management": [
            "Scout the crop regularly and monitor whether the spots spread.",
            "Remove and appropriately dispose of infected plant residues.",
            "Avoid using seed collected from infected crops.",
            "Maintain appropriate crop spacing and field sanitation.",
            "Ask a local agricultural expert whether a registered "
            "fungicide or biological option is appropriate for your crop.",
        ],
        "source": (
            "Tamil Nadu Agricultural University — Alternaria leaf blight",
            "https://agritech.tnau.ac.in/crop_protection/"
            "cotton_diseases/cotton_d6.html",
        ),
    },

    "Bacterial Blight": {
        "description": (
            "A bacterial cotton disease that may cause water-soaked "
            "or dark angular lesions and damage along leaf veins."
        ),
        "symptoms": [
            "Small, water-soaked lesions, sometimes limited by leaf veins.",
            "Lesions may become reddish-brown or dark.",
            "Darkening of veins or veinlets may occur.",
            "Severe infection may cause leaf damage and defoliation.",
        ],
        "management": [
            "Use healthy, certified seed from a reliable source.",
            "Remove infected crop residues and volunteer cotton plants "
            "where appropriate.",
            "Clean tools and equipment before moving from affected "
            "areas to healthy areas.",
            "Follow locally recommended crop rotation and sanitation practices.",
            "Consult a local agricultural expert before deciding whether "
            "a disease-control product is appropriate.",
        ],
        "source": (
            "Tamil Nadu Agricultural University — Bacterial blight",
            "https://agritech.tnau.ac.in/crop_protection/"
            "cotton_diseases/cotton_d9.html",
        ),
    },

    "Fusarium Wilt": {
        "description": (
            "A soil-associated fungal disease that can cause wilting, "
            "yellowing and discoloration of the plant's vascular tissue."
        ),
        "symptoms": [
            "Wilting that may be more noticeable during warm conditions.",
            "Yellowing or drying of lower leaf margins.",
            "Brown discoloration may occur in the lower stem or root.",
            "Affected plants may show reduced growth or die.",
        ],
        "management": [
            "Use healthy planting seed and varieties recommended "
            "for your local area.",
            "Avoid moving potentially infested soil on tools, equipment "
            "or footwear between fields.",
            "Clean equipment after working in affected field areas.",
            "Discuss crop rotation and soil-borne disease management "
            "with an agricultural extension specialist.",
            "Do not assume that crop rotation alone will eliminate "
            "the fungus from the soil.",
        ],
        "source": (
            "UC IPM — Fusarium wilt in cotton",
            "https://ipm.ucanr.edu/agriculture/cotton/fusarium-wilt/",
        ),
    },

    "Healthy Leaf": {
        "description": (
            "The model classified the uploaded image as a healthy "
            "cotton leaf. This does not rule out early or hidden disease."
        ),
        "symptoms": [
            "The model did not assign the image to one of the four "
            "disease categories.",
            "A healthy prediction is not a laboratory diagnosis.",
        ],
        "management": [
            "Continue regular scouting of leaves, stems and the whole crop.",
            "Use healthy seed and follow locally recommended crop practices.",
            "Monitor unusual yellowing, spots, wilting or changes in growth.",
            "Upload a clear image if you notice new symptoms.",
        ],
        "source": None,
    },

    "Verticillium Wilt": {
        "description": (
            "A soil-associated fungal disease that may cause wilting, "
            "yellowing between leaf veins and vascular discoloration."
        ),
        "symptoms": [
            "Interveinal yellowing followed by damaged leaf tissue.",
            "Wilting or premature leaf fall may occur.",
            "Brown vascular discoloration may occur in stems or branches.",
            "Symptoms can overlap with Fusarium wilt and other plant problems.",
        ],
        "management": [
            "Inspect affected plants and note how symptoms are distributed "
            "across the field.",
            "Use locally recommended tolerant varieties where available.",
            "Discuss suitable crop rotation with an agricultural expert.",
            "Avoid transporting infested soil and plant residue to clean fields.",
            "Seek field or laboratory confirmation when wilt symptoms "
            "are severe or difficult to distinguish.",
        ],
        "source": (
            "UC IPM — Verticillium wilt in cotton",
            "https://ipm.ucanr.edu/agriculture/cotton/verticillium-wilt/",
        ),
    },
}


# =====================================================
# 3. LOAD SAVED MODEL
# =====================================================

@st.cache_resource
def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = load_model(
        str(MODEL_PATH),
        custom_objects={
            "preprocess_input": preprocess_input
        },
        compile=False,
    )

    if model.input_shape[1:] != (224, 224, 3):
        raise ValueError("Unexpected model input shape.")

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError("Model output does not match the five classes.")

    return model


# =====================================================
# 4. WEBPAGE HEADER
# =====================================================

st.title("🌿 Cotton Disease Detection and Management")

st.write(
    "Upload a cotton leaf image to get a model prediction, "
    "symptom information and general management guidance."
)

st.caption(
    "Deep learning: MobileNetV2 | "
    "Five cotton leaf categories | 224 × 224 input"
)

st.divider()


# =====================================================
# 5. IMAGE UPLOAD AND PREDICTION
# =====================================================

uploaded_file = st.file_uploader(
    "Upload a cotton leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:

    try:
        original_image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Uploaded Leaf")
            st.image(
                original_image,
                caption="Input image",
                use_container_width=True,
            )

        with right:
            st.subheader("Disease Prediction")

            if st.button(
                "Detect Disease",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner("Analyzing the image..."):

                    model = get_model()

                    resized = original_image.resize(
                        (224, 224),
                        Image.Resampling.BILINEAR,
                    )

                    image_array = np.asarray(
                        resized,
                        dtype=np.float32,
                    )

                    image_batch = np.expand_dims(
                        image_array,
                        axis=0,
                    )

                    # The trained model already contains
                    # MobileNetV2 preprocessing internally.
                    scores = model.predict(
                        image_batch,
                        verbose=0,
                    )[0]

                    predicted_index = int(np.argmax(scores))
                    disease_name = CLASS_NAMES[predicted_index]
                    confidence = float(scores[predicted_index])

                    st.session_state["prediction"] = {
                        "disease": disease_name,
                        "confidence": confidence,
                        "scores": scores.tolist(),
                    }

        # Display the result outside the button block so that
        # it remains visible after Streamlit reruns the page.
        result = st.session_state.get("prediction")

        if result is not None:
            disease_name = result["disease"]
            confidence = result["confidence"]

            st.divider()
            st.header("Prediction Result")

            result_col, confidence_col = st.columns(2)

            with result_col:
                st.subheader(disease_name)

            with confidence_col:
                st.metric(
                    "Model confidence",
                    f"{confidence * 100:.2f}%",
                )

            if confidence < 0.60:
                st.warning(
                    "The model's confidence is relatively low. "
                    "Try a clear image of one leaf and seek expert "
                    "confirmation before acting on the result."
                )

            info = DISEASE_INFO[disease_name]

            st.subheader("Disease Information")
            st.write(info["description"])

            with st.expander("View symptoms to check", expanded=True):
                for symptom in info["symptoms"]:
                    st.markdown(f"- {symptom}")

            with st.expander("Management and prevention", expanded=True):
                for step, recommendation in enumerate(
                    info["management"], start=1
                ):
                    st.markdown(f"{step}. {recommendation}")

            if info["source"] is not None:
                st.subheader("Agricultural Reference")
                source_title, source_url = info["source"]
                st.markdown(f"[{source_title}]({source_url})")

            st.subheader("Scores Across the Five Classes")

            score_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Score (%)": (
                    np.array(result["scores"], dtype=float) * 100
                ),
            }).set_index("Class")

            st.bar_chart(score_df)

            st.info(
                "Important: This is an experimental image classifier, "
                "not a confirmed diagnosis. Symptoms can overlap, and "
                "the model only knows its five trained categories. "
                "Management guidance is general. Confirm the disease "
                "and any pesticide decision with a qualified local "
                "agricultural expert and current product-label guidance."
            )

    except Exception as error:
        st.error("The image could not be analyzed.")
        st.exception(error)

else:
    st.info("Upload a JPG or PNG cotton leaf image to begin.")

st.divider()
st.caption(
    "Cotton Disease AI | MobileNetV2 | Agriculture Analytics"
)
