import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os

# =====================
# CONFIG
# =====================
MODEL_PATH = "hand_landmarker.task"

# Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found at {MODEL_PATH}")
    print("Please download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    exit()

# Initialize MediaPipe Hand Landmarker (New Tasks API)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

CLASS_NAMES = ["rock", "paper", "scissors"]

# Hand Connections for drawing (standard MediaPipe Hands topology)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),             # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),             # Index
    (5, 9), (9, 10), (10, 11), (11, 12),        # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),      # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),      # Pinky
    (0, 17)                                     # Palm base
]

def draw_landmarks_manual(image, landmarks, connections):
    """Manually draw landmarks and connections if solutions.drawing_utils is unavailable."""
    h, w, _ = image.shape
    # Draw connections
    for connection in connections:
        p1 = landmarks[connection[0]]
        p2 = landmarks[connection[1]]
        cv2.line(image, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (0, 255, 0), 2)
    
    # Draw points
    for lm in landmarks:
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)

def get_hand_move(landmarks):
    """
    Logic-based move detection using hand landmarks.
    Indexes: 4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
    """
    # 1. Detect extended fingers
    # A finger is 'open' if the tip (e.g., 8) is higher than the joint below (e.g., 6)
    finger_tips = [8, 12, 16, 20]
    finger_states = []
    
    # Check 4 fingers
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            finger_states.append(1) # Open
        else:
            finger_states.append(0) # Closed
            
    # Check thumb (uses X coordinate for horizontal open check)
    # Using landmark 3 as base joint for tip 4
    if landmarks[4].x < landmarks[3].x:
        thumb_open = 1
    else:
        thumb_open = 0
        
    num_open = sum(finger_states)
    
    # 2. Map to RPS
    if num_open == 0:
        return "rock"
    elif num_open == 4 or (num_open == 3 and thumb_open == 1):
        return "paper"
    elif num_open == 2 and finger_states[0] == 1 and finger_states[1] == 1:
        return "scissors"
    
    return "unknown"

# =====================
# GAME LOGIC
# =====================
def decide_winner(p1, p2):
    if p1 == p2: return "Tie"
    rules = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
    if p1 == "unknown" or p2 == "unknown": return "Waiting..."
    return "Player 1" if rules.get(p1) == p2 else "Player 2"

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(0)

player1_score = 0
player2_score = 0
ROUND_TIME = 3
last_round_time = time.time()
state = "game" # game -> result

p1_move, p2_move = "unknown", "unknown"
result_text = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Convert image to MediaPipe Format
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # Process Frame (Video Mode requires timestamp)
    timestamp_ms = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, timestamp_ms)
    
    current_moves = {"left": "unknown", "right": "unknown"}
    
    if results.hand_landmarks:
        for i, hand_lms in enumerate(results.hand_landmarks):
            # Draw landmarks
            draw_landmarks_manual(frame, hand_lms, HAND_CONNECTIONS)
            
            # Identify side by X coord of wrist (landmark 0)
            cx = int(hand_lms[0].x * w)
            side = "left" if cx < w // 2 else "right"
            
            move = get_hand_move(hand_lms)
            current_moves[side] = move

    # ---------- Game Logic ----------
    now = time.time()
    elapsed = now - last_round_time
    
    if state == "game":
        countdown = max(0, ROUND_TIME - int(elapsed))
        p1_live = current_moves["left"]
        p2_live = current_moves["right"]
        
        if elapsed >= ROUND_TIME:
            p1_move, p2_move = p1_live, p2_live
            winner = decide_winner(p1_move, p2_move)
            if winner == "Player 1": player1_score += 1
            elif winner == "Player 2": player2_score += 1
            result_text = f"{winner} Wins!" if "Player" in winner else winner
            state = "result"
            last_round_time = now
    else:
        if elapsed >= 2: # Show result for 2 seconds
            state = "game"
            last_round_time = now

    # ---------- UI ----------
    cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 1)
    
    # Left Move
    cv2.putText(frame, f"P1: {current_moves['left'].upper()}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Right Move
    cv2.putText(frame, f"P2: {current_moves['right'].upper()}", (w//2 + 20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Scores
    cv2.putText(frame, f"{player1_score} : {player2_score}", (w//2 - 50, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    if state == "game":
        cv2.putText(frame, str(countdown if countdown > 0 else "GO!"), (w//2 - 40, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
    else:
        cv2.putText(frame, result_text, (w//2 - 150, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        cv2.putText(frame, f"{p1_move} vs {p2_move}", (w//2 - 100, h//2 + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow("RPS MediaPipe - New API", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

detector.close()
cap.release()
cv2.destroyAllWindows()
