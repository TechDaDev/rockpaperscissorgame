# ✊ Rock Paper Scissors AI - Real-time Hand Gesture Game

A premium, AI-powered Rock-Paper-Scissors game built with **Streamlit**, **MediaPipe**, and **OpenCV**. This application uses cutting-edge computer vision to track hand landmarks in real-time and determine game moves automatically.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0078D4?logo=google&logoColor=white)

## ✨ Features

*   **Premium Dark UI**: A modern, glassmorphic design with neon accents and fluid animations.
*   **Real-time Hand Tracking**: Powered by MediaPipe's Hand Landmarker for high-precision gesture recognition.
*   **Smart Game Logic**:
    *   **Pulsing "GO!" Phase**: Automatically waits for hands to appear.
    *   **Stable Capture**: 1-second stabilization timer before locking in the move.
    *   **Local Multiplayer**: Split-screen detection for two players.
*   **High Performance**: Optimized 720p processing with model warm-up to prevent initial lag.
*   **Live Scoreboard**: Real-time score tracking that updates without refreshing the camera feed.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit (with Custom CSS)
*   **Computer Vision**: MediaPipe Hand Landmarker (Tasks API)
*   **Video Handling**: `streamlit-webrtc`
*   **Logic**: Python, OpenCV, NumPy

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TechDaDev/rockpaperscissorgame.git
   cd rockpaperscissorgame
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install streamlit streamlit-webrtc mediapipe opencv-python av numpy
   ```

4. **Prepare the model**:
   Ensure `hand_landmarker.task` is in the root directory. You can download it from [Google's MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models).

## 🎮 How to Play

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```
2. **Setup**: Allow camera access and wait for "🚀 Loading AI Models..." to finish.
3. **Gameplay**:
    *   **Player 1** stands on the LEFT side.
    *   **Player 2** stands on the RIGHT side.
    *   Wait for the countdown (3... 2... 1...) and show your move when **GO!** appears.
    *   The game will wait for your hands to be steady for 1 second before deciding the winner.

## 📁 Project Structure

*   `app.py`: The main Streamlit web application.
*   `hand_landmarker.task`: MediaPipe pre-trained model for hand tracking.
*   `step1-5_*.py`: Development scripts used for training, testing, and prototyping different parts of the AI pipeline.

## 🤝 Contributing

Feel free to fork this project, open issues, or submit pull requests to improve the game UX or logic!

---
Developed by [TechDaDev](https://github.com/TechDaDev)
