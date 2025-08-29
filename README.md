# Emotion Detection from Faces

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
📁 Emotion-Detection-Capstone
│── app.py # Streamlit webapp (UI + Integration)
│── detector.py # Core detection & emotion analysis logic
│── requirements.txt # Project dependencies


---

## 🎯 Features
✅ Real-time emotion detection from webcam  
✅ Spoof detection (only real faces accepted in webcam mode)  
✅ Image upload support without spoof-check  
✅ Bounding box detection using Haarcascade  
✅ Optimized processing (frame skipping + error handling)  
✅ Streamlit-based user-friendly web interface  

---

## ⚙️ How It Works
1. **Face Detection** → Haarcascade locates faces.  
2. **Emotion Analysis** → DeepFace predicts emotions.  
3. **Anti-Spoofing** → Enabled only in webcam mode.  
4. **UI** → Streamlit provides two tabs:  
   - **🎥 Webcam (Live)** → Real-time detection with spoof-check.  
   - **📷 Image Upload** → Emotion detection for uploaded photos.  

---

## 🚀 Setup & Run

### 1️⃣ Install dependencies
pip install -r requirements.txt

### 2️⃣ Run locally
streamlit run app.py

### 🌐 Streamlit App Link
https://emotion-detection-using-deepface-with-anti-spoofing-a6mbcfnr8x.streamlit.app/
