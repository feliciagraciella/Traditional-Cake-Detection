import streamlit as st
from PIL import Image
import numpy as np

import torch
import cv2

import os
import sys
sys.path.insert(0, './yolov7')

from torchvision.transforms import functional as F
from models.yolo import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

label_colors = {
    'cenil': (255, 0, 0),   # Red
    'cucur': (0, 150, 0),   # Green
    'dadar-gulung': (0, 0, 255),   # Blue
    'getuk-lindri': (252, 161, 3), # Yellow
    'lumpur': (255, 0, 255), # Magenta
    'putri-salju': (0, 150, 255), # Cyan
    'serabi': (128, 0, 128), # Purple
    'wajik': (128, 128, 0)  # Olive
    # Add more labels and colors as needed
}

# Function to load YOLOv7 model
def load_model(weights_path='traditional-cake-model/yolov7-v4.pt'):
    model = attempt_load(weights_path, map_location=torch.device('cpu'))
    return model

def make_square(img):
    width, height = img.size
    new_size = max(width, height)
    new_img = Image.new("RGB", (new_size, new_size), (0, 0, 0))
    paste_position = ((new_size - width) // 2, (new_size - height) // 2)
    new_img.paste(img, paste_position)
    return new_img

# Function to perform object detection
def detect_objects(model, img_path, conf_thres=0.001, iou_thres=0.7):
    img0 = Image.open(img_path)
    if img0.mode != 'RGB':
        img0 = img0.convert('RGB')
    
    new_image = make_square(img0)
    img = F.resize(new_image, [640,640])

    img_tensor = F.to_tensor(img).unsqueeze(0)

    with torch.no_grad():  # Disable gradient tracking to avoid the error
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

    img_np = np.array(img)

    # Extracting labels, confidence scores, and bounding boxes
    labels = pred[:, -1].cpu().numpy().astype(int)
    confidences = pred[:, 4].cpu().numpy()
    boxes = pred[:, :4].cpu().numpy()

    return labels, confidences, boxes, img_np

    
# Streamlit UI
st.title('Traditional Cake Detection')

my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

col1, col2= st.columns(2)

with col1:
    st.subheader('Uploaded Image')
    
    if my_upload is not None:
        image = Image.open(my_upload)

        st.image(image, caption='Uploaded Image', use_column_width=True)
    else:
        st.image('images/dadargulung3.jpg')

with col2:
    st.subheader('Detection')
    
    if my_upload is not None:
        model = load_model()

        labels, confidences, boxes, processed_img = detect_objects(model, my_upload, conf_thres=0.1)

        classes = ['cenil', 'cucur', 'dadar-gulung', 'getuk-lindri', 'lumpur', 'putri-salju', 'serabi', 'wajik']

        for label, conf, box in zip(labels, confidences, boxes):
            if conf > 0.8:
                class_name = classes[int(label)]
                color = label_colors.get(class_name, (255, 255, 255)) 
                
                plot_one_box(box, processed_img, label=f'{class_name}: {conf:.2f}', line_thickness=2, color=color)

        st.image(processed_img, caption='Processed Image', use_column_width=True)

        st.text("Top 3 Classes : ")

        # i = 0
        # for label, conf, box in zip(labels, confidences, boxes):
        #     i += 1
        #     class_name = classes[int(label)]
        #     st.text(f'{i}. {class_name}: {conf:.2f}')

        printed_labels = set()
        printed_count = 0
        
        for label, conf, box in zip(labels, confidences, boxes):
            class_name = classes[int(label)]
            
            # Check if the label has already been printed
            if class_name not in printed_labels:
                # Print the label, confidence, and increment the count
                printed_labels.add(class_name)
                printed_count += 1
                st.text(f'{printed_count}. {class_name}: {conf:.2f}')
                
                # Break the loop if the top 3 labels have been printed
                if printed_count == 3:
                    break
            
    else:
        st.image('runs/detect/yolov7-traditionalcake-detect7/dadargulung3.jpg')


# st.sidebar.header('Upload an image')
