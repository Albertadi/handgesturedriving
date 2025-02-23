# Hand Gesture Recognition with MLP & Mediapipe

This repository contains a **hand gesture recognition project** using **Mediapipe**, **TensorFlow (MLP model)**, and **OpenCV** for real-time gesture control. It includes **data preprocessing**, **model training**, and **live gesture detection**.

## Project Structure

| File | Description |
|------|------------|
| **CSCI218 Group Project.ipynb** | Jupyter Notebook for data preprocessing, model training, and evaluation |
| **gesture_mlp_model_random_rotation.h5** | Trained MLP model for gesture classification |
| **save_dataset.py** | Script to preprocess and save the dataset from images to csv of landmark coordinates |
| **scaler.npy** | Stored mean values for feature normalization |
| **scaler_scale.npy** | Stored scaling factors for feature normalization |

---

## Features
**Real-time gesture recognition** using OpenCV and Mediapipe  
**MLP model** trained to classify hand gestures  
**Confusion matrix & evaluation reports**  

---

## Setup Instructions

### **Install Dependencies**
Ensure you have the required Python packages installed:
```bash
pip install tensorflow opencv-python mediapipe numpy scikit-learn joblib matplotlib seaborn
```
Run the last block within **CSCI218 Group Project.ipynb** to launch the system.
