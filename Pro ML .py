# ============================================================
# CROP RECOMMENDATION SYSTEM
# Random Forest Classifier
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "Test Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Number of rows:", len(df))


# ============================================================
# 2. REMOVE UNNECESSARY COLUMN
# ============================================================

if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)


# ============================================================
# 3. REMOVE MISSING VALUES
# ============================================================

df = df.dropna()


# ============================================================
# 4. SELECT INPUT FEATURES
# ============================================================

X = df[
    [
        "N",
        "P",
        "K",
        "pH",
        "rainfall",
        "temperature"
    ]
]


# Target
y = df["Crop"]


# ============================================================
# 5. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.14,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================================
# 8. TEST MODEL
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("        MODEL PERFORMANCE")
print("======================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 9. SAVE MODEL
# ============================================================

joblib.dump(model, "crop_recommendation_model.pkl")

print("\nModel saved successfully!")


# ============================================================
# 10. USER INPUT
# ============================================================

print("\n")
print("======================================")
print("      CROP RECOMMENDATION SYSTEM")
print("======================================")

print("\nEnter the following values:\n")


try:

    # Nitrogen
    N = float(input("Enter Nitrogen (N): "))

    # Phosphorus
    P = float(input("Enter Phosphorus (P): "))

    # Potassium
    K = float(input("Enter Potassium (K): "))

    # Soil pH
    pH = float(input("Enter Soil pH: "))

    # Rainfall
    rainfall = float(input("Enter Rainfall: "))

    # Temperature
    temperature = float(input("Enter Temperature: "))


    # ========================================================
    # 11. CREATE USER INPUT DATAFRAME
    # ========================================================

    user_input = pd.DataFrame(
        {
            "N": [N],
            "P": [P],
            "K": [K],
            "pH": [pH],
            "rainfall": [rainfall],
            "temperature": [temperature]
        }
    )


    # ========================================================
    # 12. PREDICT CROP
    # ========================================================

    prediction = model.predict(user_input)

    predicted_crop = prediction[0]


    # ========================================================
    # 13. DISPLAY RESULT
    # ========================================================

    print("\n======================================")
    print("       CROP RECOMMENDATION")
    print("======================================")

    print("\nInput values:")
    print("Nitrogen    :", N)
    print("Phosphorus  :", P)
    print("Potassium   :", K)
    print("Soil pH     :", pH)
    print("Rainfall    :", rainfall)
    print("Temperature :", temperature)

    print("\n--------------------------------------")

    print("Recommended Crop:", predicted_crop)

    print("--------------------------------------")


except ValueError:

    print("\nInvalid input!")
    print("Please enter numbers only.")

import joblib

joblib.dump(model, "crop_recommendation_model.pkl")

print("Model saved successfully!")
