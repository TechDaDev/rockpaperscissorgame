import cv2
import numpy as np
import tensorflow as tf

# =====================
# CONFIG
# =====================
MODEL_PATH = "rps_mobilenet.keras"
IMG_SIZE = 224
CLASS_NAMES = ["paper", "rock", "scissors"]

SMOOTH_N = 5  # prediction smoothing window

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
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, 0)

# =====================
# CAMERA
# =====================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

pred_buffer = []

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # mirror view (more natural for user)
    frame = cv2.flip(frame, 1)

    x = preprocess(frame)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))

    pred_buffer.append(pred)
    if len(pred_buffer) > SMOOTH_N:
        pred_buffer.pop(0)

    # majority vote smoothing
    smooth_pred = max(set(pred_buffer), key=pred_buffer.count)

    label = CLASS_NAMES[smooth_pred]
    conf = probs[smooth_pred]

    # =====================
    # DRAW UI
    # =====================
    text = f"{label.upper()}  {conf:.2f}"
    
    # Draw shadow for better readability
    cv2.putText(frame, text, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.rectangle(frame, (10,10), (350,70), (0,255,0), 2)

    cv2.imshow("RPS Live", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Program closed.")
