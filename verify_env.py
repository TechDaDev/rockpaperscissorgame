import os
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "rps_mobilenet.keras"

print(f"TensorFlow version: {tf.__version__}")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file {MODEL_PATH} not found!")
    exit(1)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_processed = tf.keras.applications.mobilenet_v2.preprocess_input(dummy_img.astype(float))
    img_batch = np.expand_dims(img_processed, 0)
    
    # Predict
    preds = model.predict(img_batch, verbose=0)
    print(f"✅ Prediction test successful: {preds}")
    print(f"Classes: ['paper', 'rock', 'scissors']")
    print(f"Top class: {['paper', 'rock', 'scissors'][np.argmax(preds[0])]}")
    
except Exception as e:
    print(f"❌ Error during verification: {e}")
    import traceback
    traceback.print_exc()
