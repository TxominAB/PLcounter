import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from utils import draw_bboxes

# Load YOLOv8m model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="PL counter model", layout="wide")

st.title("PL counter model")
st.write("Upload an image to detect and count shrimps.")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    # Run inference
    results = model(img_np)
    
    # Extract bounding boxes
    detected_img, total_objects = draw_bboxes(img_np, results)

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader(f"Detected Objects: {total_objects}")
        st.image(detected_img, use_column_width=True)

    st.write(f"Total objects detected: **{total_objects}**")