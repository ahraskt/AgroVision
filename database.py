
import os
import mysql.connector
from dotenv import load_dotenv

# Get project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env file
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH, override=True)


def get_db_connection():

    host = os.getenv("TIDB_HOST")
    port = os.getenv("TIDB_PORT", "4000")
    user = os.getenv("TIDB_USER")
    password = os.getenv("TIDB_PASSWORD")
    database = os.getenv("TIDB_DATABASE")
    ca_path = os.getenv("TIDB_CA_PATH")

    # Check required settings
    if not all([host, user, password, database, ca_path]):
        raise ValueError(
            "Missing database settings. Check your .env file."
        )

    # Check CA certificate
    if not os.path.isfile(ca_path):
        raise FileNotFoundError(
            f"CA certificate not found: {ca_path}"
        )

    # Connect to TiDB Cloud
    conn = mysql.connector.connect(
        host=host.strip(),
        port=int(port),
        user=user.strip(),
        password=password,
        database=database,
        ssl_ca=ca_path,
        ssl_verify_cert=True,
        ssl_verify_identity=True,
        connection_timeout=15
    )

    return conn
