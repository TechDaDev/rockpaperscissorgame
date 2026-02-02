import cv2
import numpy as np
import os
from pathlib import Path
import random

DATA_DIR = Path("data")
CLASSES = ["paper", "rock", "scissors"]
IMG_SIZE = (224, 224)

def create_verification_grid():
    rows = []
    for c in CLASSES:
        class_folder = DATA_DIR / c
        files = list(class_folder.glob("*"))
        random.shuffle(files)
        
        # Pick 5 random images from this class
        row_imgs = []
        for i in range(min(5, len(files))):
            img = cv2.imread(str(files[i]))
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                # Draw class name on image
                cv2.putText(img, c.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                row_imgs.append(img)
        
        if row_imgs:
            rows.append(np.hstack(row_imgs))
    
    if rows:
        grid = np.vstack(rows)
        cv2.imwrite("class_verification.png", grid)
        print("✅ Created class_verification.png. Please check if labels match the hand signs.")
    else:
        print("❌ No images found in data folders.")

if __name__ == "__main__":
    create_verification_grid()
