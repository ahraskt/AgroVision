
from database import get_db_connection


def save_crop_prediction(
    nitrogen,
    phosphorus,
    potassium,
    ph,
    rainfall,
    temperature,
    predicted_crop,
    confidence
):

    connection = None
    cursor = None

    try:
        # Connect to TiDB Cloud
        connection = get_db_connection()

        cursor = connection.cursor()

        # Insert crop prediction into database
        query = """
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
            nitrogen,
            phosphorus,
            potassium,
            ph,
            rainfall,
            temperature,
            predicted_crop,
            confidence
        )

        cursor.execute(query, values)

        connection.commit()

        return cursor.lastrowid

    except Exception:
        if connection:
            connection.rollback()
        raise

    finally:
        if cursor:
            cursor.close()

        if connection:
            connection.close()
