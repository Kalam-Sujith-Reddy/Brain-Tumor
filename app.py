# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Brain Tumor Detection using CNN-CatBoost")

# Load Pre-trained ML Model
model_path = Path(settings.PT_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Upload Section
st.subheader("Upload an Image for Tumor Detection")

uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp"))

if uploaded_file is not None:
    uploaded_image = PIL.Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input Image**")
        st.image(uploaded_image, caption="Uploaded Image", width=400)

    with col2:
        if st.button("Detect Tumor"):
            try:
                res = model.predict(uploaded_image)
                res_plotted = res[0].plot()[:, :, ::-1]
                st.markdown("**Detected Image**")
                st.image(res_plotted, caption="Detected Image", width=400)
            except Exception as e:
                st.error("Tumor detection failed.")
                st.error(e)
