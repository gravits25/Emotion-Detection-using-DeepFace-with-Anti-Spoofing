import cv2
import numpy as np
from deepface import DeepFace
from typing import Tuple, Dict

# Loading Haarcascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# For realtime-webcam emotion detection
def analyze_emotion(frame: np.ndarray, frame_count: int, last_label: str = "Waiting...") -> Tuple[np.ndarray, str]:

    if frame is None or frame.size == 0:
        return frame, "No frame"

    # Resizing input to fixed size (320x240) for consistency and efficient memory utilization
    resized = cv2.resize(frame, (240, 320))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Detect faces in frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    label = last_label

    # Running deepface for every 5th frame for efficient cpu utilization
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

                    # Use detected emotion or mark as "Spoof Detected!"

                    emotion = face.get("dominant_emotion", "Unknown") if is_real else "Spoof Detected!"
                    label = emotion

                    color = (0, 255, 0) if is_real else (0, 0, 255)

                    # Drawing bounding box and label
                    cv2.rectangle(resized, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(resized, emotion, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception as e:
            label = f"{str(e)}"
    else:
        # Draw last label only
        if label[0] == 'S':
            color = (0,0,255)
        else:
            color = (0,255,0)

        # Drawing bounding box with last label

        for (x, y, w, h) in faces:
            cv2.rectangle(resized, (x, y), (x + w, y + h), color, 2)
            cv2.putText(resized, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return resized, label



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

                color = (0, 255, 0)  # always green (since spoofing not checked)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img, summary

    except Exception as e:
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (5 + 480, 40), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        cv2.putText(img, f"Error: {str(e)}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        return img, {"error": str(e)}
