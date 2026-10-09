
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

DATASET_DIR = Path(r"D:\Project\PlantDiseaseDataset")

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = Path(r"D:\Project\plant_disease_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "best_model.keras"
CLASS_PATH = OUTPUT_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --------------------------------------------------
# 2. FIND ALL IMAGES AND THEIR LABELS
# --------------------------------------------------

def find_images(folder):
    """Find images recursively inside a folder."""
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_label(image_path, root_dir):
    """
    Example:
    train/Apple/Apple Scab/image.jpg
    becomes Apple___Apple Scab
    """
    relative_parts = image_path.relative_to(root_dir).parts

    if len(relative_parts) >= 3:
        plant = relative_parts[-3]
        disease = relative_parts[-2]
        return f"{plant}___{disease}"

    if len(relative_parts) == 2:
        # Supports a test folder with one class folder level.
        return relative_parts[-2]

    # A directly stored image has no label folder.
    return None


train_files = find_images(TRAIN_DIR)
valid_files = find_images(VALID_DIR)
test_files = find_images(TEST_DIR)

if not train_files:
    raise ValueError(f"No training images found in {TRAIN_DIR}")

if not valid_files:
    raise ValueError(f"No validation images found in {VALID_DIR}")

# Build class names from training folders.
class_names = sorted({
    get_label(path, TRAIN_DIR) for path in train_files
})

if None in class_names:
    raise ValueError("Check your train folder structure.")

class_to_index = {
    name: index for index, name in enumerate(class_names)
}

num_classes = len(class_names)

print("\nNumber of training images:", len(train_files))
print("Number of validation images:", len(valid_files))
print("Number of classes:", num_classes)

print("\nClasses learned by the model:")
for index, name in enumerate(class_names):
    print(index, ":", name)

# Check that validation folders match the training classes.
for path in valid_files:
    label = get_label(path, VALID_DIR)
    if label not in class_to_index:
        raise ValueError(
            f"Validation class '{label}' is missing from train."
        )

# Save class names in the exact order used by the model.
with open(CLASS_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)


# --------------------------------------------------
# 3. LOAD AND PREPROCESS IMAGES
# --------------------------------------------------

def load_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)

    # Read the image as a 3-channel RGB image.
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])

    # Resize every image to 224 x 224 pixels.
    image = tf.image.resize(image, IMAGE_SIZE)

    # MobileNetV2 preprocessing: pixel range becomes -1 to 1.
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


def make_dataset(files, root_dir, shuffle=False):
    paths = []
    labels = []

    for path in files:
        name = get_label(path, root_dir)

        if name not in class_to_index:
            raise ValueError(
                f"Unknown class '{name}' for image {path}"
            )

        paths.append(str(path))
        labels.append(class_to_index[name])

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 2000),
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = make_dataset(
    train_files, TRAIN_DIR, shuffle=True
)

valid_ds = make_dataset(
    valid_files, VALID_DIR
)


# --------------------------------------------------
# 4. LOAD PRE-TRAINED MOBILENETV2
# --------------------------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pre-trained layers initially.
base_model.trainable = False


# --------------------------------------------------
# 5. ADD OUR OWN CLASSIFICATION LAYERS
# --------------------------------------------------

inputs = tf.keras.Input(shape=(224, 224, 3))

# Small random changes help the model generalize.
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.05)(x)

# Extract features using MobileNetV2.
x = base_model(x, training=False)

# Convert the feature maps into a feature vector.
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Reduce overfitting.
x = tf.keras.layers.Dropout(0.25)(x)

# Predict one of our plant disease classes.
outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)


# --------------------------------------------------
# 6. COMPILE THE MODEL
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# --------------------------------------------------
# 7. TRAIN AND SAVE THE BEST MODEL
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save the final model state too.
# ModelCheckpoint has already saved the best model.
model.save(OUTPUT_DIR / "final_model.keras")


# --------------------------------------------------
# 8. PLOT ACCURACY AND LOSS
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("MobileNetV2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("MobileNetV2 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_graphs.png")
plt.show()


# --------------------------------------------------
# 9. EVALUATE OR PREDICT TEST IMAGES
# --------------------------------------------------

best_model = tf.keras.models.load_model(MODEL_PATH)

# If the test folder contains class subfolders,
# evaluate accuracy and create a confusion matrix.
test_labels = [
    get_label(path, TEST_DIR) for path in test_files
]

has_test_labels = (
    len(test_files) > 0
    and all(label in class_to_index for label in test_labels)
)

if has_test_labels:
    test_ds = make_dataset(test_files, TEST_DIR)

    test_loss, test_accuracy = best_model.evaluate(test_ds)

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    probabilities = best_model.predict(test_ds)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.array([
        class_to_index[label] for label in test_labels
    ])

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    ))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(18, 18))
    display.plot(
        ax=ax,
        xticks_rotation="vertical",
        colorbar=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.show()

elif test_files:
    # Flat test folder: no labels are available,
    # so predictions can be produced, but test accuracy
    # cannot be calculated automatically.
    print(
        "\nTest images have no class labels. "
        "Generating predictions instead of test accuracy."
    )

    for path in test_files:
        image_bytes = tf.io.read_file(str(path))
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(
            image
        )

        prediction = best_model.predict(
            tf.expand_dims(image, axis=0),
            verbose=0
        )[0]

        best_index = int(np.argmax(prediction))

        print(
            f"{path.name} -> {class_names[best_index]} "
            f"({prediction[best_index] * 100:.2f}%)"
        )

else:
    print("No test images found.")

print("\nTraining and testing steps finished.")
print("Best model:", MODEL_PATH)
print("Class names:", CLASS_PATH)