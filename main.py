import streamlit as st
from PIL import Image
import numpy as np
from torchvision.models import resnet50
import torch
from torch import nn
from torchvision import transforms

model = resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model.load_state_dict(torch.load('models/model_2_acc_0.99.pth',  map_location=torch.device('cpu')))
model.eval()

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    output = model(transform(image).unsqueeze(0))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.text('Predicted: {}'.format(np.argmax(output.detach().numpy())))