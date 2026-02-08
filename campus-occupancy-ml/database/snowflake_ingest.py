import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

# ------------------------------------------------
# Load environment variables
# ------------------------------------------------

load_dotenv()

SNOW_USER = os.getenv("SNOW_USER")
SNOW_PASS = os.getenv("SNOW_PASS")
SNOW_ACCOUNT = os.getenv("SNOW_ACCOUNT")
SNOW_WH = os.getenv("SNOW_WH")
SNOW_DB = os.getenv("SNOW_DB")
SNOW_SCHEMA = os.getenv("SNOW_SCHEMA")

# ------------------------------------------------
# Connect to Snowflake
# ------------------------------------------------

def connect():
    return snowflake.connector.connect(
        user=SNOW_USER,
        password=SNOW_PASS,
        account=SNOW_ACCOUNT,
        warehouse=SNOW_WH,
        database=SNOW_DB,
        schema=SNOW_SCHEMA
    )

# ------------------------------------------------
# Create table
# ------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS CLASS_SESSIONS (
    DATE STRING,
    TIME STRING,
    SCHOOL STRING,
    VENUE STRING,
    VENUE_CAPACITY INTEGER,
    ENROLLED_STUDENTS INTEGER,
    ACTUAL_ATTENDANCE INTEGER
)
"""

def create_table():
    conn = connect()
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    conn.close()
    print("✅ Table ready")

# ------------------------------------------------
# Upload CSV to Snowflake
# ------------------------------------------------

def upload_csv(csv_path):

    df = pd.read_csv(csv_path)

    conn = connect()
    cur = conn.cursor()

    for _, row in df.iterrows():
        values = tuple(row.values)
        cur.execute("""
            INSERT INTO CLASS_SESSIONS
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, values)

    conn.commit()
    conn.close()

    print("✅ Data uploaded")

# ------------------------------------------------
# Main pipeline
# ------------------------------------------------

if __name__ == "__main__":

    create_table()

    csv_file = "../sample_data/meow_sample.csv"
    upload_csv(csv_file)

    print("🚀 Snowflake ingestion complete")
