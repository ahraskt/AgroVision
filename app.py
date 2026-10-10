
"""
AgroVision Flask application
- Crop recommendation
- Plant disease detection using MobileNetV2
- MySQL storage for crop prediction history
- Admin login, history and logout

Before deployment, set the database and admin environment variables in Render.
Do not put database passwords or real secrets in this file.
"""

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
    Flask, render_template, request, session, redirect, url_for
)
from werkzeug.utils import secure_filename


# ------------------------------------------------------------
# 1. PATHS: use paths relative to this app.py file, not D:\...
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

CROP_MODEL_PATH = BASE_DIR / "crop_recommendation_model.pkl"
DISEASE_MODEL_PATH = BASE_DIR / "plant_disease_model" / "best_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "plant_disease_model" / "class_names.json"

app = Flask(__name__)

# Set SECRET_KEY in Render environment variables for deployment.
app.secret_key = os.environ.get("SECRET_KEY", "change-this-for-local-development")

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # Maximum upload: 8 MB
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Admin credentials are configured through environment variables.
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")


# ------------------------------------------------------------
# 2. LOAD EACH MACHINE-LEARNING MODEL ONCE
# ------------------------------------------------------------
if not CROP_MODEL_PATH.is_file():
    raise FileNotFoundError(f"Crop model file was not found: {CROP_MODEL_PATH}")

if not DISEASE_MODEL_PATH.is_file():
    raise FileNotFoundError(f"Disease model file was not found: {DISEASE_MODEL_PATH}")

if not CLASS_NAMES_PATH.is_file():
    raise FileNotFoundError(f"Class names file was not found: {CLASS_NAMES_PATH}")

crop_model = joblib.load(CROP_MODEL_PATH)
disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)

with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
    class_names = json.load(file)


def allowed_file(filename):
    """Return True only for JPG, JPEG or PNG filenames."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ------------------------------------------------------------
# 3. MYSQL CONNECTION
# ------------------------------------------------------------
def get_db_connection():
    """
    Connect to the hosted MySQL database using environment variables.

    Configure DB_HOST, DB_USER, DB_PASSWORD, DB_NAME and DB_PORT
    in Render. Never use localhost for a hosted database unless the
    database really runs inside the same service/container.
    """
    required = ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing database environment variable(s): " + ", ".join(missing)
        )

    return mysql.connector.connect(
        host=os.environ["DB_HOST"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        port=int(os.environ.get("DB_PORT", "3306")),
        connection_timeout=10,
    )


# ------------------------------------------------------------
# 4. HOME PAGE
# ------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ------------------------------------------------------------
# 5. CROP RECOMMENDATION + SAVE PREDICTION TO MYSQL
# ------------------------------------------------------------
@app.route("/crop", methods=["GET", "POST"])
def crop():
    recommended_crop = None
    confidence = None
    error = None
    database_warning = None

    if request.method == "POST":
        try:
            # Read the form fields. The HTML input names must match these.
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            pH = float(request.form["pH"])
            rainfall = float(request.form["rainfall"])
            temperature = float(request.form["temperature"])

            # Keep feature names and order consistent with model training.
            user_input = pd.DataFrame([{
                "N": N,
                "P": P,
                "K": K,
                "pH": pH,
                "rainfall": rainfall,
                "temperature": temperature,
            }])

            # Predict the crop.
            recommended_crop = str(crop_model.predict(user_input)[0])

            # Some classifiers support predict_proba; not all do.
            if hasattr(crop_model, "predict_proba"):
                probabilities = crop_model.predict_proba(user_input)
                confidence = round(float(np.max(probabilities[0])) * 100, 2)

            # Save prediction to the hosted MySQL database.
            connection = None
            cursor = None
            try:
                connection = get_db_connection()
                cursor = connection.cursor()
                sql = """
                    INSERT INTO crop_predictions
                    (nitrogen, phosphorus, potassium, ph, rainfall,
                     temperature, predicted_crop, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    N, P, K, pH, rainfall, temperature,
                    recommended_crop, confidence
                ))
                connection.commit()
            except Exception:
                app.logger.exception("Crop was predicted, but saving to MySQL failed")
                database_warning = (
                    "Crop prediction completed, but the result could not be saved "
                    "to the database. Please check the hosted database settings."
                )
            finally:
                if cursor is not None:
                    cursor.close()
                if connection is not None and connection.is_connected():
                    connection.close()

        except (ValueError, KeyError):
            error = "Please enter valid numerical values in all required fields."
        except Exception:
            app.logger.exception("Crop prediction failed")
            error = "Could not make a crop prediction. Please check the input and model."

    return render_template(
        "crop.html",
        recommended_crop=recommended_crop,
        confidence=confidence,
        error=error,
        database_warning=database_warning,
    )


# ------------------------------------------------------------
# 6. PLANT DISEASE DETECTION
# ------------------------------------------------------------
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
                image_bytes = uploaded_file.read()
                image = tf.keras.utils.load_img(
                    io.BytesIO(image_bytes),
                    target_size=(224, 224),
                    color_mode="rgb",
                )
                image_array = tf.keras.utils.img_to_array(image)
                image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                    image_array
                )
                image_array = np.expand_dims(image_array, axis=0)

                probabilities = disease_model.predict(image_array, verbose=0)[0]
                best_index = int(np.argmax(probabilities))

                if best_index >= len(class_names):
                    raise ValueError("Model output count does not match class_names.json")

                prediction = str(class_names[best_index])
                confidence = round(float(probabilities[best_index]) * 100, 2)

                # Dataset class names commonly look like Apple___Apple_scab.
                if "___" in prediction:
                    plant_name, disease_name = prediction.split("___", 1)
                else:
                    plant_name = "Unknown"
                    disease_name = prediction

                safe_name = secure_filename(uploaded_file.filename) or "leaf_image.jpg"
                unique_name = f"{uuid.uuid4().hex}_{safe_name}"
                save_path = UPLOAD_FOLDER / unique_name
                save_path.write_bytes(image_bytes)

                image_url = url_for("static", filename=f"uploads/{unique_name}")

            except Exception:
                app.logger.exception("Plant disease prediction failed")
                error = (
                    "Could not process this image. Please upload a valid leaf photo."
                )

    return render_template(
        "disease.html",
        prediction=prediction,
        plant_name=plant_name,
        disease_name=disease_name,
        confidence=confidence,
        image_url=image_url,
        error=error,
    )


# ------------------------------------------------------------
# 7. ABOUT PAGE
# ------------------------------------------------------------
@app.route("/about")
def about():
    return render_template("about.html")


# ------------------------------------------------------------
# 8. ADMIN LOGIN
# ------------------------------------------------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session.clear()
            session["admin_logged_in"] = True
            return redirect(url_for("admin_history"))

        error = "Invalid username or password."

    return render_template("admin_login.html", error=error)


# ------------------------------------------------------------
# 9. ADMIN HISTORY: requires admin login
# ------------------------------------------------------------
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
            SELECT id, nitrogen, phosphorus, potassium, ph, rainfall,
                   temperature, predicted_crop, confidence, created_at
            FROM crop_predictions
            ORDER BY created_at DESC
        """)
        predictions = cursor.fetchall()
        return render_template("admin_history.html", predictions=predictions)

    except Exception:
        app.logger.exception("Could not load admin prediction history")
        return (
            "Could not load prediction history. Check the database environment "
            "variables, database access permissions, and crop_predictions table.",
            500,
        )
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


# ------------------------------------------------------------
# 10. ADMIN LOGOUT
# ------------------------------------------------------------
@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_login"))


# ------------------------------------------------------------
# 11. LOCAL DEVELOPMENT
# On Render, Gunicorn starts the app using: gunicorn app:app
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)

