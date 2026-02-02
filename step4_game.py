import cv2
import numpy as np
import tensorflow as tf
import time

# =====================
# CONFIG
# =====================
MODEL_PATH = "rps_mobilenet.keras"
IMG_SIZE = 224
CLASS_NAMES = ["paper", "rock", "scissors"]

# ROI Settings (Defining squares where players should put their hands)
ROI_SIZE = 300  # Size of the square box
ROI_OFFSET_X = 50  # Distance from the edge/center
ROI_OFFSET_Y = 100 # Vertical position from top

# =====================
# LOAD MODEL
# =====================
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# =====================
# PREPROCESS
# =====================
def preprocess(roi):
    # Ensure ROI is valid
    if roi.size == 0:
        return None
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(float))
    return np.expand_dims(img, 0)

# =====================
# GAME LOGIC
# =====================
def decide_winner(p1, p2):
    if p1 == p2:
        return "Tie"

    rules = {
        "rock": "scissors",
        "scissors": "paper",
        "paper": "rock"
    }

    if rules[p1] == p2:
        return "Player 1"
    else:
        return "Player 2"

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# =====================
# GAME STATE
# =====================
player1_score = 0
player2_score = 0

ROUND_TIME = 3
FREEZE_TIME = 2

last_round_time = time.time()
state = "countdown"
countdown_value = ROUND_TIME

p1_move = None
p2_move = None
p1_conf = 0.0
p2_conf = 0.0
result_text = ""

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ---------- Define ROI Squares ----------
    # P1 Box (Left)
    p1_x1, p1_y1 = ROI_OFFSET_X, ROI_OFFSET_Y
    p1_x2, p1_y2 = p1_x1 + ROI_SIZE, p1_y1 + ROI_SIZE
    
    # P2 Box (Right)
    p2_x1, p2_y1 = w - ROI_OFFSET_X - ROI_SIZE, ROI_OFFSET_Y
    p2_x2, p2_y2 = p2_x1 + ROI_SIZE, p2_y1 + ROI_SIZE

    # Crop ROIs
    roi1 = frame[p1_y1:p1_y2, p1_x1:p1_x2]
    roi2 = frame[p2_y1:p2_y2, p2_x1:p2_x2]

    # ---------- predictions ----------
    def predict_roi(roi):
        x = preprocess(roi)
        if x is None: return "...", 0.0
        probs = model.predict(x, verbose=0)[0]
        pred = np.argmax(probs)
        return CLASS_NAMES[pred], probs[pred]

    if state == "countdown":
        p1_label, p1_live_conf = predict_roi(roi1)
        p2_label, p2_live_conf = predict_roi(roi2)
    else:
        p1_label = p1_move if p1_move else "..."
        p2_label = p2_move if p2_move else "..."
        p1_live_conf = p1_conf
        p2_live_conf = p2_conf

    now = time.time()
    elapsed = now - last_round_time

    # ---------- state machine ----------
    if state == "countdown":
        countdown_value = max(0, ROUND_TIME - int(elapsed))
        if elapsed >= ROUND_TIME:
            state = "freeze"
            last_round_time = now

    elif state == "freeze":
        p1_move, p1_conf = predict_roi(roi1)
        p2_move, p2_conf = predict_roi(roi2)
        
        winner = decide_winner(p1_move, p2_move)

        if winner == "Player 1":
            player1_score += 1
        elif winner == "Player 2":
            player2_score += 1

        result_text = f"{winner} wins" if winner != "Tie" else "Tie"
        state = "result"
        last_round_time = now

    elif state == "result":
        if elapsed >= FREEZE_TIME:
            state = "countdown"
            p1_move = None
            p2_move = None
            last_round_time = now

    # ---------- drawing UI ----------
    
    # Draw Background Boxes for ROIs
    cv2.rectangle(frame, (p1_x1, p1_y1), (p1_x2, p1_y2), (0, 255, 0), 2)
    cv2.rectangle(frame, (p2_x1, p2_y1), (p2_x2, p2_y2), (0, 255, 0), 2)
    
    # Labels with Confidence
    p1_text = f"P1: {p1_label.upper()} ({p1_live_conf:.2f})"
    p2_text = f"P2: {p2_label.upper()} ({p2_live_conf:.2f})"
    
    cv2.putText(frame, p1_text, (p1_x1, p1_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, p2_text, (p2_x1, p2_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Scoreboard
    score_text = f"{player1_score} : {player2_score}"
    cv2.putText(frame, score_text, (w//2 - 60, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # Center Overlays
    if state == "countdown":
        color = (0, 0, 255) if countdown_value > 1 else (0, 255, 255)
        cv2.putText(frame, str(countdown_value if countdown_value > 0 else "GO!"),
            (w//2 - 40, h//2),
            cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

    if state == "result":
        cv2.putText(frame, result_text, (w//2 - 140, h//2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 50, 50), 3)
        
        moves_text = f"{p1_move} vs {p2_move}"
        cv2.putText(frame, moves_text, (w//2 - 100, h//2 + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow("RPS Duel - ROI Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Game closed.")
