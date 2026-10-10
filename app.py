from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CROP_MODEL_PATH = BASE_DIR / "crop_recommendation_model.pkl"

DISEASE_MODEL_PATH = (
    BASE_DIR / "plant_disease_model" / "best_model.keras"
)

CLASS_NAMES_PATH = (
    BASE_DIR / "plant_disease_model" / "class_names.json"
)
import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from flask import (
    request,
    render_template,
    redirect,
    url_for
)
from werkzeug.utils import secure_filename


from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import joblib
import mysql.connector

app = Flask(__name__)
app.secret_key = "crop_project_secret_key_2026"

# ==========================================
# PLANT DISEASE MODEL
# ==========================================

MODEL_PATH = Path(
    r"D:\Project\plant_disease_model\best_model.keras"
)

CLASS_PATH = Path(
    r"D:\Project\plant_disease_model\class_names.json"
)

disease_model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)
UPLOAD_FOLDER = Path(app.root_path) / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Load the trained model once when Flask starts.
disease_model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )

# ============================================================
# ADMIN LOGIN DETAILS
# ============================================================

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# ============================================================
# LOAD ML MODEL
# ============================================================
import joblib

crop_model = joblib.load(CROP_MODEL_PATH)
model = joblib.load("crop_recommendation_model.pkl")


# ============================================================
# MYSQL CONNECTION
# ============================================================

def get_db_connection():

    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Ahras@2002",
        database="crop_project"
    )

    return connection


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# CROP RECOMMENDATION
# ============================================================

@app.route("/crop", methods=["GET", "POST"])
def crop():

    recommended_crop = None
    confidence = None
    error = None

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # GET USER INPUT
            # ------------------------------------------------

            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            pH = float(request.form["pH"])
            rainfall = float(request.form["rainfall"])
            temperature = float(request.form["temperature"])


            # ------------------------------------------------
            # CREATE INPUT DATAFRAME
            # ------------------------------------------------

            user_input = pd.DataFrame({
                "N": [N],
                "P": [P],
                "K": [K],
                "pH": [pH],
                "rainfall": [rainfall],
                "temperature": [temperature]
            })


            # ------------------------------------------------
            # PREDICT CROP
            # ------------------------------------------------

            prediction = model.predict(user_input)

            recommended_crop = prediction[0]


            # ------------------------------------------------
            # MODEL CONFIDENCE
            # ------------------------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(user_input)

                confidence = round(
                    max(probabilities[0]) * 100,
                    2
                )


            # ------------------------------------------------
            # SAVE DATA TO MYSQL
            # ------------------------------------------------

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
                N,
                P,
                K,
                pH,
                rainfall,
                temperature,
                str(recommended_crop),
                confidence
            )

            cursor.execute(sql, values)

            connection.commit()

            cursor.close()
            connection.close()


        except ValueError:

            error = "Please enter valid numerical values."


        except mysql.connector.Error as e:

            error = "Database error: " + str(e)


        except Exception as e:

            error = "Error: " + str(e)


    return render_template(
        "crop.html",
        recommended_crop=recommended_crop,
        confidence=confidence,
        error=error
    )



# ==========================================
# PLANT DISEASE DETECTION
# ==========================================

@app.route("/disease", methods=["GET", "POST"])
def disease():

    prediction = None
    plant_name = None
    disease_name = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":

        # 1. Check whether an image was uploaded
        uploaded_file = request.files.get("leaf_image")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please select a leaf image."

        elif not allowed_file(uploaded_file.filename):
            error = "Please upload a JPG, JPEG, or PNG image."

        else:
            try:
                # 2. Read the image using TensorFlow
                image_bytes = uploaded_file.read()

                image = image = tf.keras.utils.load_img(
                    io.BytesIO(image_bytes),
                    target_size=(224, 224),
                    color_mode="rgb"
                    )

                # 3. Convert the image into a numerical array
                image_array = tf.keras.utils.img_to_array(image)

                # 4. Apply MobileNetV2 preprocessing
                image_array = (
                    tf.keras.applications.mobilenet_v2.preprocess_input(
                        image_array
                    )
                )

                # 5. Add batch dimension
                image_array = np.expand_dims(image_array, axis=0)

                # 6. Predict the class
                probabilities = disease_model.predict(
                    image_array, verbose=0
                )[0]

                best_index = int(np.argmax(probabilities))
                prediction = class_names[best_index]
                confidence = round(
                    float(probabilities[best_index]) * 100, 2
                )

                # 7. Separate plant name and disease name
                if "___" in prediction:
                    plant_name, disease_name = prediction.split(
                        "___", 1
                    )
                else:
                    plant_name = "Unknown"
                    disease_name = prediction

                # 8. Save the uploaded image for displaying the result
                safe_name = secure_filename(uploaded_file.filename)
                if not safe_name:
                    safe_name = "leaf_image.jpg"

                # Use a unique filename to avoid overwriting uploads.
                import uuid
                unique_name = f"{uuid.uuid4().hex}_{safe_name}"
                save_path = UPLOAD_FOLDER / unique_name

                # Re-save the original uploaded bytes.
                save_path.write_bytes(image_bytes)

                image_url = url_for(
                    "static",
                    filename=f"uploads/{unique_name}"
                )

            except Exception as exc:
                app.logger.exception("Plant disease prediction failed")
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
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template("about.html")


# ============================================================
# ADMIN LOGIN
# ============================================================

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():

    error = None

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:

            session["admin_logged_in"] = True

            return redirect(url_for("admin_history"))

        else:

            error = "Invalid username or password."

    return render_template(
        "admin_login.html",
        error=error
    )

# ============================================================
# ADMIN HISTORY
# ============================================================

@app.route("/admin/history")
def admin_history():

    # Check whether admin is logged in

    if not session.get("admin_logged_in"):

        return redirect(url_for("admin_login"))


    connection = None
    cursor = None

    try:

        # Connect to MySQL

        connection = get_db_connection()

        cursor = connection.cursor(dictionary=True)


        # Get all prediction records

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


    except mysql.connector.Error as e:

        return f"Database error: {e}"


    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# ADMIN LOGOUT
# ============================================================

@app.route("/admin/logout")
def admin_logout():

    session.pop("admin_logged_in", None)

    return redirect(url_for("admin_login"))

# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False
    )