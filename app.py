import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import torch

st.set_page_config(page_title="AbyssScan - Sonar Debris Detection", layout="wide")
st.title("AbyssScan - AI Sonar Debris Detection")
st.write("Upload a side-scan sonar image to detect debris, see why it was flagged, and how urgent it is.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

def get_severity(confidence, box_width, box_height, image_width, image_height):
    box_area_ratio = (box_width * box_height) / (image_width * image_height)
    if confidence < 0.4:
        return "Low", "Needs human review (low confidence)"
    elif box_area_ratio > 0.05 and confidence >= 0.7:
        return "High", "Large object, high confidence, likely significant hazard"
    elif confidence >= 0.7:
        return "Medium", "Confirmed object, moderate size"
    else:
        return "Low", "Small or uncertain detection"

def make_whole_image_heatmap(pt_model, img):
    img_resized = cv2.resize(img, (416, 416))
    rgb_img = img_resized.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float()

    activations = []
    def hook(module, input, output):
        activations.append(output)
    target_layer = pt_model.model[-2]
    handle = target_layer.register_forward_hook(hook)
    with torch.no_grad():
        pt_model(input_tensor)
    handle.remove()

    act = activations[0][0]
    heatmap = act.mean(dim=0).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return overlay

uploaded_file = st.file_uploader("Upload a sonar image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Sonar Image", use_container_width=True)

    results = model.predict(source=img, conf=0.25, imgsz=416)
    r = results[0]
    annotated = r.plot()
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_container_width=True)

    if len(r.boxes) == 0:
        st.info("No debris detected in this image.")
    else:
        heatmap_img = make_whole_image_heatmap(model.model, img)
        st.image(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB), caption="Why the AI flagged this (heatmap)", use_container_width=True)

        st.subheader("Detections")
        for box in r.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            w, h = float(x2 - x1), float(y2 - y1)
            img_h, img_w = r.orig_shape
            severity, reason = get_severity(conf, w, h, img_w, img_h)
            st.write("Confidence: " + str(round(conf, 2)) + " | Severity: " + severity + " | " + reason)
