import os
import psycopg2

# Establish a connection to the PostgreSQL database
# The DATABASE_URL environment variable is provided by Replit
DATABASE_URL = os.environ['DATABASE_URL']

# Connect to the database
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# Create a table to store the work log data
cur.execute("""
    CREATE TABLE IF NOT EXISTS work_log (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        data JSON NOT NULL
    )
""")

# Commit the changes and close the connection
conn.commit()
cur.close()
conn.close()
