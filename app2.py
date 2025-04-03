import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Custom CSS for background color
st.markdown(
    """
    <style>
    body {
        background-color: #0071BF;
        color: white;
    }
    .stApp {
        background-color: #0071BF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the custom YOLOv8 model
model_path = "best.pt"  # Update this path to your model
model = YOLO(model_path)

st.title("Grobest Group PL counter app")
st.write("Upload an image for PL counting.")

# Display image from local drive at the top
local_image_path = "GB_logo.png"  # Update this path to your local image
top_image = Image.open(local_image_path)
st.image(top_image, caption='Top Image', use_column_width=True)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Run YOLOv8 model inference
    results = model(opencv_image)

    # Get the number of detected objects (assuming one class for detection)
    num_objects = len(results[0].boxes)

    # Draw bounding boxes on the image
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Access the coordinates correctly
        cv2.rectangle(opencv_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # Convert back to PIL image
    result_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))

    st.image(result_image, caption='Processed Image', use_column_width=True)
    st.write(f"Total shrimps detected: {num_objects}")
    
    # Input field for Total sample weight (g)
    total_weight = st.number_input("Total sample weight (g)", min_value=0.0, format="%.2f")
    previous_weight = st.number_input("Previous sampling weight (g)", min_value=0.0, format="%.2f")
    previous_sampling_days = st.number_input("Number of days since last sampling", min_value=1.0, format="%f")

    if total_weight > 0 and num_objects > 0:
        # Calculate Average Body Weight (g)
        average_body_weight = total_weight / num_objects
        growth_rate = (average_body_weight - previous_weight) / previous_sampling_days
        st.write(f"Average Body Weight (g): {average_body_weight:.2f}")
        st.write(f"Growth rate (g/day): {growth_rate:.2f}")
    else:
        st.write("Please ensure the total sample weight is greater than 0 and objects are detected.")
