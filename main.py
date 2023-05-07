import streamlit as st
from PIL import Image
import numpy as np
from torchvision.models import resnet50
import torch
from torch import nn
from torchvision import transforms


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.load_state_dict(torch.load('models/model_2_acc_0.99.pth'))
    model.eval()


    print(model(transform(image).unsqueeze(0)))
    with torch.no_grad():
        st.image(
            image,
            caption="uploaded image",
            use_column_width=True,
        )