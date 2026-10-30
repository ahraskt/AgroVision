
from database import get_db_connection


def save_disease_prediction(
    image_name,
    image_data,
    plant_name,
    disease_name,
    confidence
):
    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        query = """
            INSERT INTO disease_predictions
            (image_name, image_data, plant_name, disease_name, confidence)
            VALUES (%s, %s, %s, %s, %s)
        """

        cursor.execute(
            query,
            (
                image_name,
                image_data,
                plant_name,
                disease_name,
                confidence
            )
        )

        connection.commit()
        return cursor.lastrowid

    except Exception:
        if connection and connection.is_connected():
            connection.rollback()
        raise

    finally:
        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()
