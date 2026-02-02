import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# CONFIG
MODEL_PATH = "rps_mobilenet.keras"
TRAIN_DIR = "data_split/train"
VAL_DIR = "data_split/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 1. Load Data
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=42
)
val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=42
)

# 2. Load existing model
if not os.path.exists(MODEL_PATH):
    print("❌ Model not found. Run Step 2 first.")
    exit()

model = keras.models.load_model(MODEL_PATH)

# 3. UNFREEZE the base model
# Note: In the previous step, the base model was named something like 'mobilenetv2_1.00_224'
# We find it by its class type as well to be safe
base_model = None
for layer in model.layers:
    if "mobilenetv2" in layer.name.lower():
        base_model = layer
        break

if base_model:
    base_model.trainable = True
    # Freeze the bottom layers, unfreeze the top (head-specific) layers
    fine_tune_at = 100 
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"✅ Unfroze {len(base_model.layers) - fine_tune_at} layers of {base_model.name}")
else:
    print("⚠️ Could not find base model layer by name. Training full model.")
    model.trainable = True

# 4. Recompile with a VERY LOW learning rate
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Train
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True)
    ]
)

print(f"✅ Fine-tuning complete. Updated model saved to {MODEL_PATH}")

# 6. Evaluate and Generate Classification Report & Confusion Matrix
print("\n📊 Evaluating model on validation set...")

# Get class names
class_names = train_ds.class_names

# Collect all true labels and predictions
y_true = []
y_pred = []

print("🔄 Generating predictions...")
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=-1))

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Classification Report
print("\n📝 Classification Report:")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Confusion Matrix
print("📉 Creating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Rock Paper Scissors')
plt.tight_layout()

# Save the plot
cm_plot_path = "confusion_matrix.png"
plt.savefig(cm_plot_path)
print(f"✅ Confusion matrix saved as '{cm_plot_path}'")

# Optional: Show plot (if in an interactive environment)
# plt.show()
