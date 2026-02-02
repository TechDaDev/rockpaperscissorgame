import cv2
import numpy as np
import tensorflow as tf

# =====================
# CONFIG
# =====================
MODEL_PATH = "rps_mobilenet.keras"
IMG_SIZE = 224
CLASS_NAMES = ["paper", "rock", "scissors"]

# Load Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

def get_hand_mask(frame):
    """
    Skin color detection mask (YCrCb space)
    """
    # Convert to YCrCb
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    
    # Skin color range (tuned for typical indoor lighting)
    lower = np.array([0, 133, 77], dtype="uint8")
    upper = np.array([255, 173, 127], dtype="uint8")
    
    mask = cv2.inRange(ycrcb, lower, upper)
    
    # Cleanup mask (Erode/Dilate)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    return mask

def preprocess_for_model(roi):
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(float))
    return np.expand_dims(img, 0)

# Camera
cap = cv2.VideoCapture(0)

# ROI definition
ROI_H = 300
ROI_W = 300

print("🔍 HAND DEBUGGER MODE")
print("Place your hand in the green box.")
print("The 'MASK' window shows what the computer treats as 'Hand'.")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Center ROI
    x1, y1 = (w - ROI_W) // 2, (h - ROI_H) // 2
    x2, y2 = x1 + ROI_W, y1 + ROI_H
    
    # 1. Extract Raw ROI
    roi_raw = frame[y1:y2, x1:x2]
    
    # 2. Get Mask
    mask = get_hand_mask(roi_raw)
    
    # 3. Apply Mask (See what the model might be seeing if we isolated skin)
    roi_masked = cv2.bitwise_and(roi_raw, roi_raw, mask=mask)
    
    # 4. Predict on RAW ROI (The model was trained on raw images usually)
    # Note: We predict on roi_raw because standard models expect the full context, 
    # but we show the mask to help the user understand detection quality.
    input_tensor = preprocess_for_model(roi_raw)
    probs = model.predict(input_tensor, verbose=0)[0]
    idx = np.argmax(probs)
    label = CLASS_NAMES[idx]
    conf = probs[idx]
    
    # --- Visualization ---
    # Put label on frame
    color = (0, 255, 0) if conf > 0.8 else (0, 0, 255)
    cv2.putText(frame, f"PRED: {label.upper()} ({conf:.2f})", (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Show Probability Bar Chart
    for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
        bar_w = int(p * 200)
        cv2.rectangle(frame, (x1, y2 + 20 + i*30), (x1 + bar_w, y2 + 45 + i*30), (255, 255, 0), -1)
        cv2.putText(frame, f"{name}: {p:.2f}", (x1 + 210, y2 + 40 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Combine visuals: Original Frame + Mask + Masked ROI
    # Resize mask to fit nicely
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    sidebar = np.vstack([roi_raw, mask_rgb, roi_masked])
    
    # Resize sidebar if it's too tall for the window
    if sidebar.shape[0] > frame.shape[0]:
        scale = frame.shape[0] / sidebar.shape[0]
        sidebar = cv2.resize(sidebar, (0,0), fx=scale, fy=scale)

    cv2.imshow("Main View", frame)
    cv2.imshow("Hand Analysis (Raw / Mask / Applied)", sidebar)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
