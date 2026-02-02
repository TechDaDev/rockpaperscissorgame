import streamlit as st
import cv2
import av
import time
import numpy as np
import threading

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ======================
# PAGE CONFIG & CUSTOM CSS
# ======================

st.set_page_config(
    page_title="Rock Paper Scissors | AI Game",
    page_icon="✊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Dark Theme CSS
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Outfit', sans-serif;
    }
    
    /* Hide default header */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Main Title with Animation */
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #f953c6, #b91d73, #00d2ff, #f953c6);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: gradient-flow 3s linear infinite;
    }
    
    @keyframes gradient-flow {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }
    
    /* Score Cards with Enhanced Animations */
    .score-container {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-bottom: 2rem;
    }
    
    .score-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 1.5rem 3rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .score-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .score-card:hover::before {
        left: 100%;
    }
    
    .score-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5);
    }
    
    .score-card.p1 {
        border-left: 4px solid #00d2ff;
    }
    
    .score-card.p2 {
        border-left: 4px solid #f953c6;
    }
    
    .score-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
        font-weight: 400;
    }
    
    .score-value {
        font-size: 4rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 0 20px rgba(255,255,255,0.3);
        transition: transform 0.3s ease;
    }
    
    .score-card:hover .score-value {
        transform: scale(1.1);
    }
    
    .score-value.p1 {
        color: #00d2ff;
        text-shadow: 0 0 30px rgba(0, 210, 255, 0.5);
    }
    
    .score-value.p2 {
        color: #f953c6;
        text-shadow: 0 0 30px rgba(249, 83, 198, 0.5);
    }
    
    /* Video Container - Fully Responsive */
    .video-wrapper {
        background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 24px;
        padding: 5px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        margin: 0 auto 2rem;
        max-width: 100%;
        overflow: hidden;
        transition: box-shadow 0.3s ease;
    }
    
    .video-wrapper:hover {
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.5);
    }
    
    .video-wrapper iframe, .video-wrapper video {
        width: 100% !important;
        height: auto !important;
        border-radius: 20px;
    }
    
    [data-testid="stVideo"] {
        width: 100% !important;
    }
    
    /* Instructions Panel - Enhanced */
    .instructions {
        background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin: 2rem auto;
        max-width: 750px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    .instructions h3 {
        color: #00d2ff;
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-size: 1.5rem;
        text-shadow: 0 0 20px rgba(0, 210, 255, 0.3);
    }
    
    .instructions ul {
        color: rgba(255,255,255,0.85);
        line-height: 2;
        list-style: none;
        padding-left: 0;
    }
    
    .instructions li {
        padding: 0.5rem 0;
        transition: transform 0.2s ease;
    }
    
    .instructions li:hover {
        transform: translateX(5px);
    }
    
    .move-icon {
        font-size: 1.8rem;
        margin-right: 0.8rem;
        display: inline-block;
        transition: transform 0.3s ease;
    }
    
    .instructions li:hover .move-icon {
        transform: scale(1.2) rotate(10deg);
    }
    
    /* Button Styling with Pulse Animation */
    .stButton > button {
        background: linear-gradient(90deg, #f953c6, #b91d73);
        color: white;
        border: none;
        padding: 0.9rem 3rem;
        border-radius: 50px;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(249, 83, 198, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(249, 83, 198, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 992px) {
        .score-container {
            gap: 2rem;
        }
        .score-card {
            padding: 1.2rem 2rem;
        }
        .score-value {
            font-size: 3rem;
        }
    }
    
    @media (max-width: 768px) {
        .main-title { 
            font-size: 2.5rem; 
        }
        .subtitle {
            font-size: 1rem;
        }
        .score-container {
            gap: 1.5rem;
            flex-direction: column;
            align-items: center;
        }
        .score-card { 
            padding: 1rem 2rem;
            min-width: 200px;
        }
        .score-value { 
            font-size: 2.8rem; 
        }
        .instructions {
            padding: 1.5rem;
        }
        .instructions h3 {
            font-size: 1.2rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 1.8rem;
        }
        .score-value {
            font-size: 2.2rem;
        }
        .stButton > button {
            padding: 0.7rem 2rem;
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ======================
# CONFIG
# ======================

MODEL_PATH = "hand_landmarker.task"

CLASS_NAMES = ["rock", "paper", "scissors"]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


# ======================
# SHARED STATE
# ======================

class SharedState:
    def __init__(self):
        self.p1 = 0
        self.p2 = 0
        self.lock = threading.Lock()

if "shared_state" not in st.session_state:
    st.session_state.shared_state = SharedState()

shared_state = st.session_state.shared_state


# ======================
# MEDIAPIPE INIT (Cached)
# ======================

@st.cache_resource
def get_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Warm-up: Process a dummy image to pre-load model components
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    mp_dummy = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_img)
    detector.detect_for_video(mp_dummy, 0)
    
    return detector

with st.spinner("🚀 Loading AI Models..."):
    detector = get_detector()


# ======================
# DRAW (Enhanced)
# ======================

def draw_landmarks(image, landmarks, side="left"):
    h, w, _ = image.shape
    # Color based on player side
    line_color = (255, 210, 0) if side == "left" else (198, 83, 249)  # Cyan / Magenta
    point_color = (255, 255, 255)
    
    for c in HAND_CONNECTIONS:
        p1 = landmarks[c[0]]
        p2 = landmarks[c[1]]
        cv2.line(image,
                 (int(p1.x*w), int(p1.y*h)),
                 (int(p2.x*w), int(p2.y*h)),
                 line_color, 3)

    for lm in landmarks:
        cv2.circle(image,
                   (int(lm.x*w), int(lm.y*h)),
                   6, point_color, -1)
        cv2.circle(image,
                   (int(lm.x*w), int(lm.y*h)),
                   8, line_color, 2)


# ======================
# MOVE LOGIC
# ======================

def get_hand_move(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_states = []
    
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            finger_states.append(1)
        else:
            finger_states.append(0)
            
    if landmarks[4].x < landmarks[3].x:
        thumb_open = 1
    else:
        thumb_open = 0
        
    num_open = sum(finger_states)
    
    if num_open == 0:
        return "rock"
    elif num_open == 4 or (num_open == 3 and thumb_open == 1):
        return "paper"
    elif num_open == 2 and finger_states[0] == 1 and finger_states[1] == 1:
        return "scissors"
    
    return "unknown"


def decide_winner(p1, p2):
    if p1 == p2: return "Tie"
    rules = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
    if p1 == "unknown" or p2 == "unknown": return "Waiting..."
    return "Player 1" if rules.get(p1) == p2 else "Player 2"


def get_move_emoji(move):
    emojis = {"rock": "✊", "paper": "🖐️", "scissors": "✌️", "unknown": "❓"}
    return emojis.get(move, "❓")


# ======================
# GAME ENGINE (Callback Handler)
# ======================

class GameEngine:
    def __init__(self):
        self.last_time = time.time()
        self.round_time = 3
        self.result_pause = 2
        self.go_timeout = 3.0
        self.detection_delay = 1.0
        self.state = "countdown"
        self.p1_final_move = "unknown"
        self.p2_final_move = "unknown"
        self.result_text = ""
        self.detection_start_time = None

    def capture_results(self, current_moves, shared_state):
        self.p1_final_move = current_moves["left"]
        self.p2_final_move = current_moves["right"]
        winner = decide_winner(self.p1_final_move, self.p2_final_move)
        
        with shared_state.lock:
            if winner == "Player 1":
                shared_state.p1 += 1
            elif winner == "Player 2":
                shared_state.p2 += 1
        
        self.result_text = f"{winner} Wins!" if "Player" in winner else winner
        self.state = "result"
        self.last_time = time.time()
        self.detection_start_time = None

    def draw_premium_text(self, img, text, pos, size, color, thickness=2, glow=True):
        x, y = pos
        if glow:
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness+3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), thickness, cv2.LINE_AA)

    def process(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Translucent UI Background Labels
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (15, 15, 15), -1)
        cv2.rectangle(overlay, (0, h-60), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect_for_video(mp_img, int(time.time() * 1000))

        current_moves = {"left": "unknown", "right": "unknown"}
        if res.hand_landmarks:
            for lm in res.hand_landmarks:
                cx = int(lm[0].x * w)
                side = "left" if cx < w // 2 else "right"
                draw_landmarks(img, lm, side)
                current_moves[side] = get_hand_move(lm)

        now = time.time()
        elapsed = now - self.last_time

        if self.state == "countdown":
            countdown = max(0, self.round_time - int(elapsed))
            if elapsed >= self.round_time:
                self.state = "go"
                self.last_time = now
            
            if countdown > 0:
                # Modern Countdown UI
                center = (w // 2, h // 2)
                cv2.circle(img, center, 74, (0, 210, 255), 3, cv2.LINE_AA)
                cv2.circle(img, center, 70, (30, 30, 30), -1)
                self.draw_premium_text(img, str(countdown), (w // 2 - 25, h // 2 + 25), 3.0, (0, 210, 255), 6)
            else:
                self.draw_premium_text(img, "GO!", (w // 2 - 80, h // 2 + 25), 3.0, (0, 255, 100), 7)

        elif self.state == "go":
            hands_detected = current_moves["left"] != "unknown" or current_moves["right"] != "unknown"
            if hands_detected:
                if self.detection_start_time is None:
                    self.detection_start_time = now
                if now - self.detection_start_time >= self.detection_delay:
                    self.capture_results(current_moves, shared_state)
            
            if elapsed >= self.go_timeout:
                if self.state == "go":
                    self.capture_results(current_moves, shared_state)

            # Pulsing GO! Animation
            pulse = abs(np.sin(now * 6)) * 0.4 + 0.6
            color = (int(0*pulse), int(255*pulse), int(100*pulse))
            self.draw_premium_text(img, "GO!", (w // 2 - 80, h // 2 + 25), 3.0, color, 7)
            
            if self.detection_start_time:
                prog = min(1.0, (now - self.detection_start_time) / self.detection_delay)
                p_w = 400
                cv2.rectangle(img, (w//2 - p_w//2, h//2 + 80), (w//2 + p_w//2, h//2 + 95), (40, 40, 40), -1)
                cv2.rectangle(img, (w//2 - p_w//2, h//2 + 80), (w//2 - p_w//2 + int(p_w*prog), h//2 + 95), (0, 255, 100), -1)

        else:  # result
            if elapsed >= self.result_pause:
                self.state = "countdown"
                self.last_time = now
            
            # Result Panel Overlay
            panel_w, panel_h = 600, 200
            px, py = w//2 - panel_w//2, h//2 - panel_h//2
            sub = img[py:py+panel_h, px:px+panel_w]
            black_rect = np.zeros_like(sub)
            img[py:py+panel_h, px:px+panel_w] = cv2.addWeighted(sub, 0.4, black_rect, 0.6, 0)
            cv2.rectangle(img, (px, py), (px+panel_w, py+panel_h), (249, 83, 198), 2, cv2.LINE_AA)
            
            res_color = (0, 255, 100) if "Wins" in self.result_text else (100, 100, 255)
            self.draw_premium_text(img, self.result_text, (px + 40, py + 90), 2.1, res_color, 4)
            self.draw_premium_text(img, f"{self.p1_final_move.upper()} vs {self.p2_final_move.upper()}", 
                                (px + 40, py + 160), 1.0, (180, 180, 180), 2)

        # Standard split line and labels
        cv2.line(img, (w // 2, 0), (w // 2, h), (120, 120, 120), 1)
        self.draw_premium_text(img, f"P1: {current_moves['left'].upper()}", (20, 50), 1.0, (255, 210, 0), 2)
        self.draw_premium_text(img, f"P2: {current_moves['right'].upper()}", (w // 2 + 20, 50), 1.0, (249, 83, 198), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Persistent engine
if "game_engine" not in st.session_state:
    st.session_state.game_engine = GameEngine()

engine = st.session_state.game_engine

def video_frame_callback(frame):
    return engine.process(frame)

# ======================
# UI LAYOUT
# ======================

# Title
st.markdown('<h1 class="main-title">✊ Rock Paper Scissors</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Hand Gesture Recognition Game</p>', unsafe_allow_html=True)

# Score Display (Fragment for live updates)
@st.fragment(run_every=1.0)
def score_display():
    st.markdown(f"""
    <div class="score-container">
        <div class="score-card p1">
            <div class="score-label">Player 1</div>
            <div class="score-value p1">{shared_state.p1}</div>
        </div>
        <div class="score-card p2">
            <div class="score-label">Player 2</div>
            <div class="score-value p2">{shared_state.p2}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

score_display()

# Video Stream
st.markdown('<div class="video-wrapper">', unsafe_allow_html=True)

# More robust ICE configuration
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="rps",
    video_frame_callback=video_frame_callback,
    rtc_configuration=rtc_config,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# Instructions
st.markdown("""
<div class="instructions">
    <h3>🎮 How to Play</h3>
    <ul>
        <li><span class="move-icon">✊</span> <strong>Rock:</strong> Make a fist (all fingers closed)</li>
        <li><span class="move-icon">🖐️</span> <strong>Paper:</strong> Open hand (all fingers extended)</li>
        <li><span class="move-icon">✌️</span> <strong>Scissors:</strong> Peace sign (index + middle finger)</li>
    </ul>
    <p style="margin-top: 1rem; color: rgba(255,255,255,0.6);">
        Stand with <strong>Player 1</strong> on the left side and <strong>Player 2</strong> on the right side of the camera.
        Show your move when the countdown reaches <strong>GO!</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Reset Button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Reset Scores"):
        with shared_state.lock:
            shared_state.p1 = 0
            shared_state.p2 = 0
        st.rerun()
