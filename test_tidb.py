
from database import get_db_connection

try:

    conn = get_db_connection()

    print("Successfully connected to TiDB Cloud!")

    cursor = conn.cursor()

    cursor.execute("SELECT DATABASE()")

    result = cursor.fetchone()

    print("Connected Database:", result[0])

    cursor.execute("SHOW TABLES")

    tables = cursor.fetchall()

    print("Available Tables:")

    for table in tables:
        print(table[0])

    cursor.close()
    conn.close()

except Exception as e:

    print("Connection Failed!")
    print("Error:", e)
