import streamlit as st
from PIL import Image

from torchvision.models import mni

uploaded_file = st.file_uploader("Choose a file")def
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    