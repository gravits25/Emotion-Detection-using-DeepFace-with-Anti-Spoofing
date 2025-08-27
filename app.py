# app.py
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from detector import analyze_emotion, analyze_image


st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("🎭 Emotion Detection – Image & Realtime (with Anti-Spoofing)")

tabs = st.tabs(["🎥 Webcam (Live)", "📷 Upload Image"])

# --------------------- Tab 1: Image Upload ---------------------
with tabs[1]:
    st.subheader("Upload an image")
    file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])
    if file is not None:
        try:
            bytes_data = file.read()
            img_array = np.frombuffer(bytes_data, np.uint8)
            bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if bgr is None:
                st.error("Could not read the image. Please try a different file.")
            else:
                annotated, summary = analyze_image(bgr)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption="Result", use_container_width=True)
                st.json(summary)
        except Exception as e:
            st.error(f"Processing error: {str(e)}")

# --------------------- Tab 2: Realtime Webcam ---------------------
with tabs[0]:
    st.subheader("Realtime detection using DeepFace with anti-spoofing")

    class EmotionProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.label = "Initializing..."

        def recv(self, frame):
            try:
                img = frame.to_ndarray(format="bgr24")
                self.frame_count += 1

                # DeepFace-based anti-spoofing + emotion
                annotated, self.label = analyze_emotion(img, self.frame_count, self.label)
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

            except Exception as e:
                img = frame.to_ndarray(format="bgr24")
                cv2.putText(img, f"Err: {str(e)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

    RTC_CONFIGURATION = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": [
                        "turn:openrelay.metered.ca:80",
                        "turn:openrelay.metered.ca:443",
                        "turn:openrelay.metered.ca:443?transport=tcp",
                    ],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
            ]
        }
    )

    webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )
