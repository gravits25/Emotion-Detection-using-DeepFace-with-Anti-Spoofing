# Emotion Detection from Faces (Capstone Project)

## 📌 Overview
This project implements a **real-time emotion detection system** using **DeepFace** and **OpenCV**, with a **Streamlit web application** for interaction.  

The system:  
- Detects faces from a webcam or uploaded images.  
- Classifies facial emotions (Happy, Sad, Angry, Neutral, etc.).  
- Uses **anti-spoofing** for live webcam (prevents photos/video replay attacks).  
- Provides an **image upload option** (without anti-spoofing).  
- Runs efficiently with optimized CPU & memory usage.  
- Deploys on **Streamlit Cloud** for free hosting.  

---

## 🛠️ Tech Stack
- **Language**: Python 3.8+  
- **Libraries**:
  - [DeepFace](https://github.com/serengil/deepface) → Emotion recognition + anti-spoofing  
  - OpenCV → Haarcascade face detection + bounding boxes  
  - Streamlit → Web UI  
  - streamlit-webrtc → Real-time webcam integration  
  - NumPy, Pillow → Image handling  

---

## 📂 Project Structure
