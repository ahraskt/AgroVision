
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tkinter import Tk, filedialog

# =================================
# 1. LOAD TRAINED CNN MODEL
# =================================

model = tf.keras.models.load_model(
    "plant_disease_model/best_model.keras",
    compile=False
)

print("CNN model loaded successfully!")

# =================================
# 2. LOAD CLASS NAMES
# =================================

with open("class_names.json", "r") as file:
    class_names = json.load(file)

print("Number of classes:", len(class_names))

# =================================
# 3. SELECT IMAGE FROM COMPUTER
# =================================

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select a Plant Leaf Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png")
    ]
)

root.destroy()

if not image_path:
    print("No image selected!")
    exit()

print("\nSelected image:", image_path)

# =================================
# 4. OPEN AND RESIZE IMAGE
# =================================

image = Image.open(image_path).convert("RGB")

image = image.resize((224, 224))

# =================================
# 5. CONVERT TO NUMPY ARRAY
# =================================

image_array = np.array(image, dtype=np.float32)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# =================================
# 6. CNN PREDICTION
# =================================

predictions = model.predict(image_array, verbose=0)

predicted_index = np.argmax(predictions[0])

predicted_class = class_names[predicted_index]

confidence = predictions[0][predicted_index] * 100

# =================================
# 7. DISPLAY RESULT
# =================================

plant_name, _, disease_name = predicted_class.partition("___")

print("\n========== PREDICTION RESULT ==========")

print("Plant Name:", plant_name)
print("Disease Name:", disease_name)

print("Confidence:", round(float(confidence), 2), "%")

print("=======================================")
