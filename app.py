

import io
import json
import os
import uuid
from pathlib import Path

import joblib
import mysql.connector
import numpy as np
import pandas as pd
import tensorflow as tf

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session
)
from werkzeug.utils import secure_filename


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CROP_MODEL_PATH = BASE_DIR / "crop_recommendation_model.pkl"

DISEASE_MODEL_PATH = (
    BASE_DIR / "plant_disease_model" / "best_model.keras"
)

CLASS_NAMES_PATH = (
    BASE_DIR / "plant_disease_model" / "class_names.json"
)


# ============================================================
# 2. FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

# Set SECRET_KEY in Render environment variables before deployment.
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "local-development-key-change-before-deployment"
)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# ============================================================
# 3. LOAD CROP RECOMMENDATION MODEL
# ============================================================

if not CROP_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Crop model not found: {CROP_MODEL_PATH}"
    )

crop_model = joblib.load(CROP_MODEL_PATH)

# Use the same model variable throughout the application.
model = crop_model


# ============================================================
# 4. LOAD PLANT DISEASE MODEL
# ============================================================

if not DISEASE_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Plant disease model not found: {DISEASE_MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.is_file():
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

print("Disease model path:", DISEASE_MODEL_PATH)
print("Disease model file exists:", DISEASE_MODEL_PATH.is_file())

disease_model = tf.keras.models.load_model(
    DISEASE_MODEL_PATH
)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

print("Plant disease model loaded successfully.")


# ============================================================
# 5. ADMIN LOGIN DETAILS
# ============================================================

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")

# Configure a strong ADMIN_PASSWORD in Render.
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

if not ADMIN_PASSWORD:
    # Allows local testing, but login is disabled until configured.
    print("WARNING: Set ADMIN_PASSWORD to enable admin login.")


# ============================================================
# 6. MYSQL CONNECTION
# ============================================================

def get_db_connection():
    """
    Local development can use localhost.
    On Render, configure the database environment variables
    for a reachable hosted MySQL database.
    """

    return mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "crop_project"),
        port=int(os.environ.get("DB_PORT", "3306")),
        connection_timeout=10
    )


# ============================================================
# 7. CHECK IMAGE FILE TYPE
# ============================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# 8. HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# 9. CROP RECOMMENDATION
# ============================================================

@app.route("/crop", methods=["GET", "POST"])
def crop():
    recommended_crop = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            # Read values submitted through crop.html.
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            pH = float(request.form["pH"])
            rainfall = float(request.form["rainfall"])
            temperature = float(request.form["temperature"])

            # The column names and order should match
            # the data used to train your crop model.
            user_input = pd.DataFrame({
                "N": [N],
                "P": [P],
                "K": [K],
                "pH": [pH],
                "rainfall": [rainfall],
                "temperature": [temperature]
            })

            # Predict the suitable crop.
            prediction = model.predict(user_input)
            recommended_crop = str(prediction[0])

            # Calculate confidence if the model supports it.
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(user_input)
                confidence = round(
                    float(np.max(probabilities[0])) * 100,
                    2
                )

            # Save the prediction in MySQL.
            connection = None
            cursor = None

            try:
                connection = get_db_connection()
                cursor = connection.cursor()

                sql = """
                    INSERT INTO crop_predictions
                    (
                        nitrogen,
                        phosphorus,
                        potassium,
                        ph,
                        rainfall,
                        temperature,
                        predicted_crop,
                        confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """

                values = (
                    N, P, K, pH, rainfall, temperature,
                    recommended_crop, confidence
                )

                cursor.execute(sql, values)
                connection.commit()

            finally:
                if cursor is not None:
                    cursor.close()

                if connection is not None:
                    connection.close()

        except ValueError:
            error = "Please enter valid numerical values."

        except mysql.connector.Error:
            app.logger.exception("MySQL connection or save failed")
            error = (
                "Prediction could not be saved. "
                "Please check the database connection."
            )

        except Exception:
            app.logger.exception("Crop recommendation failed")
            error = "An error occurred while recommending the crop."

    return render_template(
        "crop.html",
        recommended_crop=recommended_crop,
        confidence=confidence,
        error=error
    )


# ============================================================
# 10. PLANT DISEASE DETECTION
# ============================================================

@app.route("/disease", methods=["GET", "POST"])
def disease():
    prediction = None
    plant_name = None
    disease_name = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("leaf_image")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please select a leaf image."

        elif not allowed_file(uploaded_file.filename):
            error = "Please upload a JPG, JPEG, or PNG image."

        else:
            try:
                # Read the uploaded image.
                image_bytes = uploaded_file.read()

                # Resize it to the model's expected input size.
                image = tf.keras.utils.load_img(
                    io.BytesIO(image_bytes),
                    target_size=(224, 224),
                    color_mode="rgb"
                )

                # Convert the image into a NumPy array.
                image_array = tf.keras.utils.img_to_array(image)

                # Preprocess the image for MobileNetV2.
                image_array = (
                    tf.keras.applications.mobilenet_v2.preprocess_input(
                        image_array
                    )
                )

                # Add the batch dimension required by the model.
                image_array = np.expand_dims(image_array, axis=0)

                # Predict the class.
                probabilities = disease_model.predict(
                    image_array,
                    verbose=0
                )[0]

                best_index = int(np.argmax(probabilities))
                prediction = class_names[best_index]

                confidence = round(
                    float(probabilities[best_index]) * 100,
                    2
                )

                # Separate the plant name and disease name.
                if "___" in prediction:
                    plant_name, disease_name = prediction.split(
                        "___", 1
                    )
                else:
                    plant_name = "Unknown"
                    disease_name = prediction

                # Save the uploaded photo for the results page.
                safe_name = secure_filename(uploaded_file.filename)

                if not safe_name:
                    safe_name = "leaf_image.jpg"

                unique_name = f"{uuid.uuid4().hex}_{safe_name}"
                save_path = UPLOAD_FOLDER / unique_name
                save_path.write_bytes(image_bytes)

                image_url = url_for(
                    "static",
                    filename=f"uploads/{unique_name}"
                )

            except Exception:
                app.logger.exception(
                    "Plant disease prediction failed"
                )
                error = (
                    "Could not process this image. "
                    "Please upload a valid leaf photo."
                )

    return render_template(
        "disease.html",
        prediction=prediction,
        plant_name=plant_name,
        disease_name=disease_name,
        confidence=confidence,
        image_url=image_url,
        error=error
    )


# ============================================================
# 11. ABOUT PAGE
# ============================================================

@app.route("/about")
def about():
    return render_template("about.html")


# ============================================================
# 12. ADMIN LOGIN
# ============================================================

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if (
            ADMIN_PASSWORD
            and username == ADMIN_USERNAME
            and password == ADMIN_PASSWORD
        ):
            session["admin_logged_in"] = True
            return redirect(url_for("admin_history"))

        error = "Invalid username or password."

    return render_template("admin_login.html", error=error)


# ============================================================
# 13. ADMIN PREDICTION HISTORY
# ============================================================

@app.route("/admin/history")
def admin_history():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                id,
                nitrogen,
                phosphorus,
                potassium,
                ph,
                rainfall,
                temperature,
                predicted_crop,
                confidence,
                created_at
            FROM crop_predictions
            ORDER BY created_at DESC
        """)

        predictions = cursor.fetchall()

        return render_template(
            "admin_history.html",
            predictions=predictions
        )

    except mysql.connector.Error:
        app.logger.exception("Could not retrieve prediction history")
        return "Could not retrieve history. Please check the database."

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# ============================================================
# 14. ADMIN LOGOUT
# ============================================================

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))


# ============================================================
# 15. RUN LOCALLY
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=False,
        use_reloader=False
    )