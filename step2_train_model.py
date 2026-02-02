import os

# Set Keras backend to torch to use GPU on Windows if possible
os.environ["KERAS_BACKEND"] = "torch"

import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# ======================
# DEVICE CHECK
# ======================
backend = keras.backend.backend()
print(f"Using Keras backend: {backend}")

gpu_available = False
if backend == "torch":
    gpu_available = torch.cuda.is_available()
    print(f"Torch GPU Available: {gpu_available}")
    if gpu_available:
        print(f"Using Device: {torch.cuda.get_device_name(0)}")
elif backend == "tensorflow":
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpus) > 0
    print(f"TensorFlow GPU Available: {gpu_available}")
    if gpu_available:
        print(f"Using GPUs: {gpus}")

if not gpu_available:
    print("⚠️ Warning: GPU not detected. Training will be slow on CPU.")
else:
    print("✅ GPU detected! Training should be fast.")

# ======================
# CONFIG
# ======================
TRAIN_DIR = "data_split/train"
VAL_DIR = "data_split/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
SEED = 42

# ======================
# DATA LOADERS
# ======================
# keras.utils.image_dataset_from_directory works across backends in Keras 3
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ======================
# PERFORMANCE OPT
# ======================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ======================
# DATA AUGMENTATION
# ======================
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.15),
])

# ======================
# BASE MODEL
# ======================
base = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base.trainable = False  # freeze backbone first

# ======================
# MODEL HEAD
# ======================
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# ======================
# COMPILE
# ======================
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# CALLBACKS
# ======================
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        "rps_mobilenet.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ======================
# TRAIN
# ======================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ======================
# PLOT
# ======================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

print("✅ Training complete. Best model saved as rps_mobilenet.keras")
