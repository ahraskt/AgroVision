import json
from pathlib import Path
from database import get_db_connection

BASE_DIR = Path(__file__).resolve().parent
JSON_FILE = BASE_DIR / "plant_disease_model" / "remedies.json"

# Load remedies from JSON
with open(JSON_FILE, "r", encoding="utf-8") as file:
    remedies_data = json.load(file)

connection = None
cursor = None

try:
    connection = get_db_connection()
    cursor = connection.cursor()

    sql = """
        INSERT INTO disease_remedies
        (
            crop_name,
            disease_name,
            pathogen,
            organic_remedies,
            chemical_treatments,
            prevention_methods
        )
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    values = []

    for class_name, remedy in remedies_data.items():

        # Extract the exact crop and disease names
        # used by your CNN model
        crop_name, disease_name = class_name.split("___", 1)

        values.append((
            crop_name,
            disease_name,
            remedy["pathogen"],
            remedy["organic"],
            remedy["chemical"],
            remedy["prevention"]
        ))

    cursor.executemany(sql, values)
    connection.commit()

    print("Remedies inserted successfully!")
    print("Total records inserted:", cursor.rowcount)

except Exception as e:
    if connection:
        connection.rollback()
    print("Database error:", e)

finally:
    if cursor:
        cursor.close()

    if connection and connection.is_connected():
        connection.close()