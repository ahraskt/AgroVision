import mysql.connector

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Ahras@2002",
    database="crop_project"
)

print("MySQL connected successfully!")

connection.close()