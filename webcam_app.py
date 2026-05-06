import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# Page config (simple design)
st.set_page_config(page_title="Object Detection", layout="centered")

st.title("🎥 Live Object Detection")
st.write("Real-time object detection using YOLOv8")

# 🔥 Simple interaction
conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()


# Video callback
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model.track(
        img,
        persist=True,
        conf=conf_threshold,
        verbose=False
    )

    result = results[0]
    annotated_frame = result.plot()

    # 🔥 Simple enhancement: object count
    count = 0
    if result.boxes is not None:
        count = len(result.boxes)

    cv2.putText(
        annotated_frame,
        f"Objects: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# Webcam section (simple layout)
st.subheader("📷 Camera Feed")

webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")

st.write("✔ YOLOv8 Active | ✔ Real-time Detection")