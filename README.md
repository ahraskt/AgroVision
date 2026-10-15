# AgroVision --- Crop Recommendation and Plant Disease Detection System

AgroVision is a web-based agriculture support project that combines
**machine learning** and **deep learning** to help users explore crop
recommendations and identify possible plant-leaf diseases from an
uploaded image.

> **Project note:** This README describes the project structure and
> features based on the repository and project details currently
> available. Before publishing, check the actual filenames, package
> versions, database table names, model input columns, and configuration
> in `app.py` so that every detail matches the latest code.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Main Features](#main-features)
-   [Technologies Used](#technologies-used)
-   [How the System Works](#how-the-system-works)
-   [Machine-Learning Crop Recommendation
    Model](#machine-learning-crop-recommendation-model)
-   [MobileNetV2 Plant Disease Model](#mobilenetv2-plant-disease-model)
-   [Database](#database)
-   [Project Structure](#project-structure)
-   [Requirements](#requirements)
-   [Installation and Setup on
    Windows](#installation-and-setup-on-windows)
-   [Running AgroVision](#running-agrovision)
-   [Testing the Features](#testing-the-features)
-   [Common Problems](#common-problems)
-   [Security and Data Notes](#security-and-data-notes)
-   [Limitations](#limitations)
-   [Future Improvements](#future-improvements)

## Project Overview

AgroVision is designed to provide two main prediction features through a
website:

1.  **Crop Recommendation:** uses numerical soil and environmental
    inputs to predict a suitable crop based on the trained crop
    recommendation model.
2.  **Plant Disease Detection:** accepts an image of a plant leaf and
    uses a trained MobileNetV2 image-classification model to predict the
    class represented by the image.

The website is built with a Python Flask backend and HTML/CSS/JavaScript
frontend. MySQL can be used to store crop prediction inputs and results
for later viewing through an admin/history feature, depending on the
database configuration.

**Important:** Machine-learning predictions are estimates based on the
training data. They are not a substitute for laboratory soil testing or
advice from an agricultural expert.

## Main Features

-   Home page introducing AgroVision.
-   Crop recommendation form.
-   Plant-leaf image upload and prediction.
-   Prediction result display, including the predicted class and
    confidence when implemented by the model and application.
-   About page explaining the project.
-   MySQL-backed prediction history and an admin view, when correctly
    configured.
-   Trained model files loaded by the Flask application; users do not
    need to retrain the models just to run predictions.

The exact availability of each feature depends on the files and
configuration in the current repository.

## Technologies Used

  -----------------------------------------------------------------------
  Technology / language               Where it is used
  ----------------------------------- -----------------------------------
  **Python**                          Main programming language for Flask
                                      routes, model loading, data
                                      processing, and prediction logic.

  **Flask**                           Backend web framework that receives
                                      browser requests, processes form
                                      submissions and image uploads,
                                      calls the models, and returns
                                      pages/results.

  **HTML5**                           Defines the structure of web pages
                                      and forms in the `templates/`
                                      directory.

  **CSS3**                            Controls page layout, colours,
                                      spacing, responsiveness, cards,
                                      buttons, and other visual styling,
                                      typically in `static/`.

  **JavaScript**                      Adds browser-side interactions
                                      where implemented, such as image
                                      preview and form behaviour.

  **scikit-learn**                    Commonly used to train or run the
                                      tabular crop recommendation model
                                      if the saved `.pkl` model was
                                      created with scikit-learn.

  **TensorFlow / Keras**              Loads and runs the trained
                                      MobileNetV2 deep-learning model for
                                      leaf-image classification.

  **NumPy**                           Numerical arrays and input/image
                                      preprocessing.

  **Pandas**                          Reads and processes tabular
                                      datasets such as CSV files during
                                      training or data preparation.

  **Pillow (PIL)**                    May be used to open, resize, or
                                      convert uploaded leaf images.

  **MySQL**                           Relational database for saving
                                      prediction inputs/results and
                                      retrieving history, if enabled in
                                      the app.

  **MySQL Connector/Python or another Python-to-MySQL connection; use the
  MySQL driver**                      specific driver imported by
                                      `app.py`.

  **Git and GitHub**                  Source-code version control and
                                      collaboration.

  **VS Code**                         Recommended editor for development;
                                      it is not required by the deployed
                                      website itself.
  -----------------------------------------------------------------------

The list above describes the likely technologies used by this project.
The installed packages and exact versions should be confirmed from
`requirements.txt` and the imports in `app.py` and the model-training
scripts.

## How the System Works

### A. Crop recommendation flow

1.  The user opens the Crop Recommendation page.
2.  The user enters the soil/environmental values requested by the form.
3.  Flask validates and prepares the values in the order expected by the
    trained model.
4.  The saved crop model predicts a crop class.
5.  The website displays the recommendation.
6.  If database history is enabled, the application saves the inputs and
    prediction to MySQL.

Use exactly the fields displayed by the current form and the same
feature order used during model training. Do not add, remove, or reorder
model inputs without updating the model and application together.

### B. Plant disease detection flow

1.  The user opens the Plant Disease page.
2.  The user selects a leaf image.
3.  Flask checks the upload and prepares the image in the format
    expected by the model.
4.  The MobileNetV2 model calculates scores for its trained output
    classes.
5.  The application selects the highest-scoring class and may display
    its confidence.
6.  The website shows the prediction.

The model can only classify categories represented in its training
classes. A prediction should be treated as a model estimate, not a
guaranteed diagnosis.

## Machine-Learning Crop Recommendation Model

### What it does

The crop recommendation model is intended for **structured/tabular
data**. Unlike the plant disease model, it does not analyse a leaf
photograph. It receives numerical features from the crop form and
predicts a crop category.

Depending on the exact model and form implementation, the inputs may
include values such as nitrogen (N), phosphorus (P), potassium (K), soil
pH, rainfall, and temperature. Confirm the exact input list and order in
the current training script and `app.py`.

### What is `crop_recommendation_model.pkl`?

The `.pkl` extension commonly indicates a Python object saved using
`pickle` or `joblib`. In this project, the file is intended to contain
the already-trained crop recommendation model.

-   **Training** is the process of learning patterns from labelled
    example data.
-   **Saving** stores the trained model in a file.
-   **Loading** restores the saved model in the Flask application.
-   **Prediction** passes a new user's input values to the loaded model.

The `.pkl` file is not the training dataset and is not the website
itself. It is a saved model artifact. It may require compatible versions
of Python and the library used to train it.

### Crop CSV files

Files such as `Train Dataset.csv` and `Test Dataset.csv` appear in the
repository. Usually:

-   A training dataset is used to fit the model.
-   A test dataset is used to evaluate performance on data not used for
    fitting.

If the Flask application loads the saved `.pkl` file directly, the
original CSV files may not be required for every prediction, but they
can be important for retraining, testing, or documenting the project.

## MobileNetV2 Plant Disease Model

### What is MobileNetV2?

**MobileNetV2** is a convolutional neural network (CNN) architecture
designed to perform image-recognition tasks efficiently. It uses
lightweight building blocks, including depthwise separable convolutions
and inverted residual connections, to reduce computation compared with
many larger image models.

This project uses MobileNetV2 for plant-leaf image classification. The
model learns visual patterns from training images and uses those
patterns to assign an uploaded image to one of its learned classes.

### How the model is used

A typical prediction process is:

1.  Load the saved Keras/TensorFlow model.
2.  Read the uploaded leaf image.
3.  Resize the image to the input dimensions expected by the model.
4.  Apply the same pixel scaling/preprocessing used during training.
5.  Run inference to obtain scores for the output classes.
6.  Map the selected output index to its class name using the class-name
    mapping, if the application uses one.
7.  Display the predicted class and, if implemented, a confidence score.

**Preprocessing matters:** image dimensions, colour channels, pixel
scaling, and class order must match the training setup. A mismatch can
cause incorrect predictions even when the model file loads successfully.

### What is the `.keras` model file?

A saved file such as `best_model.keras` (the actual filename may differ)
stores the trained neural-network architecture and learned weights in
Keras format. The application loads this file to make predictions; it
does not need to train the network again for every image.

### What is `class_names.json`?

If included, this JSON file stores the ordered class names associated
with the model's output indices. For example, a class might represent a
plant and disease combination such as `Tomato___Early_blight`, or a
healthy-leaf category. The exact labels depend on the folder names used
during training.

The class-name order must match the order used when preparing the
training dataset. Do not manually reorder the JSON labels without
verifying the training code.

### Dataset folders

The training dataset described for this project has `train`, `valid`,
and `test` folders:

-   **`train/`** --- images used to train the model.
-   **`valid/`** --- images used during training to monitor validation
    performance and tune the model.
-   **`test/`** --- separate images used to evaluate the final model, if
    the training workflow includes test evaluation.

Plant folders may include Apple, Cherry, Corn, Grape, Peach, Pepper,
Potato, Strawberry, and Tomato, with disease and healthy classes beneath
them. The precise classes depend on the dataset version included in the
project.

### Important model-file note

Make sure the actual trained model file and any required class-name file
are present where `app.py` expects them. A folder named
`plant_disease_model/` is not enough by itself unless it contains the
model and supporting files the code loads.

## Database

MySQL is intended to store prediction records so an administrator can
review previous crop recommendations. Depending on the implementation, a
record may include input values, the predicted crop, and a timestamp.

The exact database name, table names, columns, and admin authentication
implementation must be taken from the current application code or SQL
schema. This README does not assume a particular schema.

### Database setup overview

1.  Install and start MySQL Server, or provision a compatible hosted
    MySQL database.
2.  Create the database required by the application.
3.  Create the tables using the project's SQL/schema file or the
    table-creation logic in `app.py`.
4.  Configure the database host, port, database name, username, and
    password.
5.  Test a prediction and verify that the record is saved and visible in
    the history page.

Never commit real passwords, API keys, secret keys, or production
database credentials to GitHub. Use environment variables or a local
configuration file excluded by `.gitignore`.

## Project Structure

The screenshot of the repository shows items similar to the following.
The names and files may change as the project develops.

``` text
AgroVision/
├── app.py
├── Pro ML .py
├── crop_recommendation_model.pkl
├── crop_recommendation_model.py
├── Train Dataset.csv
├── Test Dataset.csv
├── plant_disease_model/
├── templates/
├── static/
├── requirements.txt          # Add if not present
├── README.md
└── .gitignore
```

Typical responsibilities:

-   **`app.py`** --- Flask application, page routes, form handling,
    model inference, image upload processing, and database operations as
    implemented.
-   **`templates/`** --- Jinja/HTML page templates rendered by Flask.
-   **`static/`** --- CSS, JavaScript, images, and other browser assets.
-   **`crop_recommendation_model.pkl`** --- saved crop recommendation
    model.
-   **`crop_recommendation_model.py` / `Pro ML .py`** --- may contain
    training or testing code; inspect the files to confirm their exact
    roles.
-   **`plant_disease_model/`** --- intended location for the trained
    image model and related class mapping, if this matches the current
    code.
-   **CSV files** --- crop data for training/testing or other data
    processing.
-   **`requirements.txt`** --- Python dependencies and preferably
    compatible version pins.
-   **`README.md`** --- project overview and setup guide.

Update this tree if your repository has a different layout or if the
trained `.keras` file is stored elsewhere.

## Requirements

For a local Windows installation, you will need:

-   Windows 10/11 or another OS supported by the project's dependencies.
-   A Python version compatible with the TensorFlow and other pinned
    package versions.
-   Git, or the ability to download the repository as a ZIP file.
-   Enough disk space and RAM to install and run TensorFlow and load the
    models.
-   MySQL Server or a configured compatible hosted database if database
    features are enabled.
-   The repository's trained model files and other required assets.

VS Code is recommended but optional. The project can also be run from a
terminal.

## Installation and Setup on Windows

### 1. Get access to the repository

The repository may be private. The owner must invite collaborators
through GitHub repository settings, and each collaborator must accept
the invitation.

### 2. Clone the repository

Open PowerShell:

``` powershell
cd Desktop
git clone https://github.com/ahraskt/AgroVision.git
cd AgroVision
```

Alternatively, use **Code → Download ZIP** on GitHub and extract it.

### 3. Open the project

``` powershell
code .
```

If the `code` command is unavailable, open VS Code manually and select
the downloaded `AgroVision` folder.

### 4. Create and activate a virtual environment

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use Command Prompt and run:

``` cmd
.venv\Scripts\activate.bat
```

### 5. Install dependencies

If `requirements.txt` exists:

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If it does not exist, the project owner should create and commit a
tested `requirements.txt` before asking collaborators to install the
project. Do not guess package versions: TensorFlow, Python, NumPy, and
scikit-learn versions can affect compatibility and saved-model loading.

### 6. Configure the database

Set up MySQL and the expected tables, then configure the connection
using the approach supported by the code. Do not copy private production
credentials into source files or GitHub.

### 7. Check the model files

Confirm that the saved crop model and the actual trained MobileNetV2
`.keras` file (plus class mapping, if required) exist at the paths
referenced by the application.

## Running AgroVision

From the project folder, with the virtual environment active:

``` powershell
python app.py
```

If Flask starts successfully, the terminal should show the local address
configured by the application, commonly:

``` text
http://127.0.0.1:5000
```

Open that address in a browser **on the same computer**. Keep the
terminal running while using the website.

The address `127.0.0.1` is local to the computer running Flask. It is
not a public internet URL. To let people use AgroVision without
installing the project, deploy it to a hosting service and configure the
model files, environment variables, and database there.

## Testing the Features

After the site starts:

1.  Open the home page and check that the navigation works.
2.  Open Crop Recommendation and submit valid values within the expected
    ranges.
3.  Check that a crop result appears and that no model-loading or
    input-shape error is reported.
4.  Open Plant Disease and upload a supported leaf image.
5.  Check that the result label matches the model's class mapping and
    that any displayed confidence is formatted correctly.
6.  If MySQL history is enabled, confirm that a prediction is saved and
    appears in the admin/history page.
7.  Test invalid inputs and unsupported file types to ensure the
    application responds safely.

## Common Problems

  ------------------------------------------------------------------------
  Problem                             What to check
  ----------------------------------- ------------------------------------
  `ModuleNotFoundError`               Activate the virtual environment and
                                      install the required package using
                                      `pip install -r requirements.txt`.

  TensorFlow installation fails       Verify the Python version and
                                      TensorFlow version are compatible
                                      with each other and the operating
                                      system.

  `FileNotFoundError` for `.pkl` or   Confirm the model file is committed
  `.keras`                            to GitHub and the path in the code
                                      matches its location.

  Crop prediction fails               Verify the number, order, and
                                      preprocessing of input features
                                      match the crop model's training
                                      setup.

  Plant prediction fails              Check image dimensions,
                                      preprocessing, class-name order, and
                                      the model path.

  MySQL connection error              Check that MySQL is running and the
                                      host, port, database, user,
                                      password, permissions, and TLS
                                      settings are correct.

  History page is empty               Check database insertion, selected
                                      database/table, and the history
                                      query.

  Port 5000 is already in use         Stop the other process using the
                                      port or configure Flask to use
                                      another available port.

  Website works locally but not for   `127.0.0.1` is local-only; use
  others                              deployment or configure network
                                      access appropriately.
  ------------------------------------------------------------------------

## Security and Data Notes

-   Do not commit passwords, secret keys, database credentials, or
    private certificates.
-   Use environment variables for secrets and provide a `.env.example`
    containing placeholder values only, if the application supports it.
-   Exclude `.env`, virtual environments, caches, temporary uploads, and
    private data from Git using `.gitignore`.
-   Validate uploaded files, limit upload size, and avoid trusting
    user-supplied filenames.
-   Protect admin routes with authentication and suitable authorization
    checks.
-   Do not expose a development Flask server directly to the public
    internet.
-   Share only datasets and model files you have permission to
    redistribute.
-   Model files saved using pickle/joblib should only be loaded from
    trusted sources.

## Limitations

-   Predictions depend on the quality, class coverage, and
    representativeness of the training data.
-   Plant disease predictions may be unreliable for blurry images,
    unusual lighting, multiple diseases, unfamiliar plants, or classes
    absent from training.
-   A displayed model confidence is not necessarily a calibrated
    probability of correctness.
-   Crop suitability also depends on local climate, soil conditions,
    season, irrigation, and other factors that may not be represented in
    the model inputs.
-   Performance and dependency compatibility depend on the computer
    running the application.

## Future Improvements

Possible future enhancements include:

-   Clear input validation and user-friendly error messages.
-   Evaluation reports and confusion matrices for both models.
-   A secure admin login and exportable prediction history.
-   Model versioning and documented training/evaluation procedures.
-   Better handling of unknown plants and unsupported images.
-   Automated tests and deployment documentation.
-   A public hosted version with secure configuration and managed
    database backups.

## Credits

Developed as an academic machine-learning and web-development project.
Update this section with the project authors, institution, dataset
attribution, and any third-party model or library acknowledgements
required by their licences.

------------------------------------------------------------------------

**Repository:** https://github.com/ahraskt/AgroVision

**Maintainer:** Add the project maintainer's preferred name or contact
details here.
