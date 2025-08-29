import cv2
import numpy as np
import re
import streamlit as st
from deepface import DeepFace
from typing import Tuple, Dict

# ---------------- Device Type Detection ---------------- #
def get_device_type() -> str:
    """
    Detects if user is on mobile or desktop using Streamlit's user-agent.
    """
    try:
        ua = st.session_state.get("user_agent", "")
        if not ua:
            # fallback: request user-agent from JS
            st.session_state.user_agent = st.experimental_user["browser"]["user_agent"]
            ua = st.session_state.user_agent

        # Simple regex check
        if re.search("Mobi|Android|iPhone", ua, re.IGNORECASE):
            return "mobile"
        else:
            return "desktop"
    except Exception:
        return "desktop"   # default fallback


# ---------------- Haarcascade Detector ---------------- #
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ---------------- Emotion Analysis ---------------- #
def analyze_emotion(frame: np.ndarray, frame_count: int, last_label: str = "Waiting...") -> Tuple[np.ndarray, str]:
    if frame is None or frame.size == 0:
        return frame, "No frame"

    # Choose resolution based on device type
    device_type = get_device_type()
    target_size = (480, 640) if device_type == "mobile" else (640, 480)
    resized = cv2.resize(frame, target_size)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    label = last_label

    # Run DeepFace only every 5th frame
    if frame_count % 5 == 0:
        try:
            result = DeepFace.analyze(
                resized,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                anti_spoofing=True
            )
            results = result if isinstance(result, list) else [result]

            for (x, y, w, h) in faces:
                for face in results:
                    is_real = face.get("is_real", True)
                    emotion = face.get("dominant_emotion", "Unknown") if is_real else "Spoof Detected!"
                    label = emotion

                    color = (0, 255, 0) if is_real else (0, 0, 255)
                    cv2.rectangle(resized, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(resized, emotion, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            label = f"{str(e)}"
    else:
        color = (0, 0, 255) if label.startswith("S") else (0, 255, 0)
        for (x, y, w, h) in faces:
            cv2.rectangle(resized, (x, y), (x + w, y + h), color, 2)
            cv2.putText(resized, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return resized, label


# ---------------- Single Image Analysis ---------------- #
def analyze_image(bgr_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Empty image")

    img = bgr_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    try:
        result = DeepFace.analyze(
            img,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        results = result if isinstance(result, list) else [result]
        summary = {}

        for (x, y, w, h) in faces:
            for face in results:
                emotion = face.get("dominant_emotion", "Unknown")
                summary = {"dominant_emotion": emotion}
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img, summary

    except Exception as e:
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (485, 40), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        cv2.putText(img, f"Error: {str(e)}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        return img, {"error": str(e)}
