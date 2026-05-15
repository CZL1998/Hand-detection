"""
One-time setup: download EMNIST Letters and train a CNN letter classifier.
First run downloads ~600 MB of data (cached afterwards).
Training takes 3-8 minutes depending on hardware.

Usage:
    python train_model.py

Output:
    letter_model.keras  (~2 MB, loaded by main.py at runtime)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds


def preprocess(image, label):
    # EMNIST images in tfds are stored transposed — fix orientation
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label - 1, tf.int32)   # labels 1–26 → 0–25
    return image, label


def load_datasets(batch=256):
    print("Loading EMNIST Letters (first run downloads ~600 MB, then cached)...")
    ds_train, ds_test = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        shuffle_files=True,
    )
    ds_train = (ds_train
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
                .shuffle(20000)
                .batch(batch)
                .prefetch(tf.data.AUTOTUNE))
    ds_test = (ds_test
               .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch)
               .prefetch(tf.data.AUTOTUNE))
    return ds_train, ds_test


def build_model():
    # Data augmentation to handle style differences between EMNIST and air-drawing
    augment = keras.Sequential([
        layers.RandomRotation(0.12),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomZoom(0.10),
    ])
    inputs = keras.Input(shape=(28, 28, 1))
    x = augment(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(26, activation="softmax")(x)
    return keras.Model(inputs, outputs)


if __name__ == "__main__":
    MODEL_PATH = "letter_model.keras"

    ds_train, ds_test = load_datasets()

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",
                                          patience=2, factor=0.5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                      patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                                        save_best_only=True, verbose=1),
    ]

    print("\nTraining...")
    model.fit(ds_train, epochs=25, validation_data=ds_test, callbacks=callbacks)

    _, acc = model.evaluate(ds_test, verbose=0)
    print(f"\nTest accuracy: {acc:.1%}")
    print(f"Model saved → {MODEL_PATH}")
