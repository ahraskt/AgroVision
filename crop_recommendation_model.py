# ============================================================
# CROP RECOMMENDATION SYSTEM
# Random Forest Classifier
# ============================================================

import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "Train Dataset.csv"

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
# DEBUG STATEMENT
# ============================================================
print("Dataset shape:", df.shape)
print("Column names:", df.columns.tolist())
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Missing values in X:", X.isnull().sum())
print("Missing values in y:", y.isnull().sum())


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
# 8. TEST MODEL AND EVALUATE PERFORMANCE
# ============================================================

# Predict crops for the test dataset
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

precision_weighted = precision_score(
    y_test, y_pred, average="weighted", zero_division=0
)

recall_weighted = recall_score(
    y_test, y_pred, average="weighted", zero_division=0
)

f1_weighted = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

# Macro averages give equal importance to every crop
precision_macro = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)

recall_macro = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)

f1_macro = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

# Display overall performance
print("\n======================================")
print("       MODEL PERFORMANCE")
print("======================================")

print(f"Accuracy          : {accuracy * 100:.2f}%")

print("\n--- Weighted Average ---")
print(f"Precision         : {precision_weighted:.4f}")
print(f"Recall            : {recall_weighted:.4f}")
print(f"F1-score          : {f1_weighted:.4f}")

print("\n--- Macro Average ---")
print(f"Precision         : {precision_macro:.4f}")
print(f"Recall            : {recall_macro:.4f}")
print(f"F1-score          : {f1_macro:.4f}")

# Detailed report for each crop
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(
    xticks_rotation=90,
    cmap="Blues",
    values_format="d"
)

plt.title("Crop Recommendation - Confusion Matrix")
plt.tight_layout()

# Save the confusion matrix without blocking input
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("Confusion matrix saved successfully!")




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