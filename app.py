import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import GANGenerator, DCGANGenerator

st.title("GAN & DCGAN FashionMNIST Generator")

# User options
model_type = st.selectbox("Select Model", ["GAN", "DCGAN"])
num_images = st.slider("Number of Images to Generate", 1, 20, 5)
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
@st.cache_resource
def load_model(model_type):
    if model_type == "GAN":
        model = GANGenerator(latent_dim=100)
        model.load_state_dict(torch.load("GAN.pt", map_location=device))
    elif model_type == "DCGAN":
        model = DCGANGenerator(latent_dim=100)
        model.load_state_dict(torch.load("DCGAN.pt", map_location=device))  # Make sure this .pt matches DCGAN architecture
    model.eval()
    return model

model = load_model(model_type)

# Generate noise
z = torch.randn(num_images, latent_dim).to(device)

# Generate images
with torch.no_grad():
    generated_images = model(z)

# Plot
fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
if num_images == 1:
    axs = [axs]

for i in range(num_images):
    axs[i].imshow(generated_images[i].cpu().squeeze().numpy(), cmap="gray")
    axs[i].axis("off")

st.pyplot(fig)
