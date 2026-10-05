import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

img_size   = (128, 128)
batch_size = 8
epochs     = 30

# ── Data generators ──────────────────────────────────────────────────────────
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode="nearest"
)

datagen_val = ImageDataGenerator(rescale=1./255)

train_data = datagen_train.flow_from_directory(
    "dataset_cls/train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=True
)

val_data = datagen_val.flow_from_directory(
    "dataset_cls/valid",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

print(f"Training samples  : {train_data.samples}")
print(f"Validation samples: {val_data.samples}")
print(f"Class indices     : {train_data.class_indices}")
# Expected: {'fake': 0, 'real': 1}

# ── Class weights (penalise fake misses more) ─────────────────────────────────
n_fake = sum(1 for f in os.listdir("dataset_cls/train/fake") if os.path.isfile(os.path.join("dataset_cls/train/fake", f)))
n_real = sum(1 for f in os.listdir("dataset_cls/train/real") if os.path.isfile(os.path.join("dataset_cls/train/real", f)))
total  = n_fake + n_real
# class 0 = fake, class 1 = real
class_weight = {
    0: (total / (2 * n_fake)) * 1.5,   # extra penalty for missing fakes
    1:  total / (2 * n_real)
}
print(f"Class weights: {class_weight}")

# ── MobileNetV2 transfer learning ────────────────────────────────────────────
base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False          # freeze base layers first

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── Phase 1: train top layers only ───────────────────────────────────────────
print("\n=== Phase 1: Training top layers ===")
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
    ModelCheckpoint("models/best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max"),
]

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weight
)

# ── Phase 2: fine-tune top 30 layers of base ─────────────────────────────────
print("\n=== Phase 2: Fine-tuning ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weight
)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("model",  exist_ok=True)
model.save("models/best_model.keras")
model.save("models/cnn_model.h5")
model.save("model/pharmaguard_model.h5")

print("\nModel training complete!")
best_acc = max(history.history["val_accuracy"] + history2.history["val_accuracy"])
print(f"Best validation accuracy: {best_acc:.4f}")

