
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

# --------------------------------------------------
# 1. LOAD THE TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = Path(
    r"D:\Project\plant_disease_model\best_model.keras"
)

CLASS_PATH = Path(
    r"D:\Project\plant_disease_model\class_names.json"
)

model = tf.keras.models.load_model(MODEL_PATH)

# Load class names in the same order used during training.
with open(CLASS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)


# --------------------------------------------------
# 2. ASK FOR A LEAF IMAGE
# --------------------------------------------------


import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select a leaf image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png")
    ]
)

if not image_path:
    print("No image selected.")
    raise SystemExit

image_path = Path(image_path)

if not image_path.is_file():
    raise FileNotFoundError(
        f"Image not found: {image_path}"
    )


# --------------------------------------------------
# 3. PREPARE THE IMAGE
# --------------------------------------------------

image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224),
    color_mode="rgb"
)

image_array = tf.keras.utils.img_to_array(image)

# Apply the same preprocessing used during training.
image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
    image_array
)

# Add a batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
image_array = np.expand_dims(image_array, axis=0)


# --------------------------------------------------
# 4. PREDICT THE DISEASE
# --------------------------------------------------

probabilities = model.predict(image_array, verbose=0)[0]

# Index of the highest predicted probability.
best_index = int(np.argmax(probabilities))

predicted_class = class_names[best_index]
confidence = float(probabilities[best_index]) * 100


# --------------------------------------------------
# 5. DISPLAY THE RESULT
# --------------------------------------------------

plant, disease = predicted_class.split("___", 1)

print("\n========== PREDICTION RESULT ==========")
print("Plant:", plant)
print("Predicted disease:", disease)
print(f"Model confidence: {confidence:.2f}%")
print("=======================================")

# Display the three highest-scoring classes.
top_indices = np.argsort(probabilities)[-3:][::-1]

'''print("\nTop 3 predictions:")

for index in top_indices:
    
    print(
        f"{class_names[index].replace('___', ' - ')}: "
        f"{probabilities[index] * 100:.2f}%"
    )'''