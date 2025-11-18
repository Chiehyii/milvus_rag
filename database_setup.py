import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
# It's better to get these from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "your_db_name")
DB_USER = os.getenv("DB_USER", "your_db_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_db_password")
TABLE_NAME = "qa_logs"

def create_database_and_table():
    """
    Connects to the PostgreSQL database and creates the qa_logs table
    if it hasn't been created yet.
    """
    conn = None
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()

        # SQL statement to create a table in PostgreSQL
        # Using SERIAL for auto-incrementing primary key
        # Using TIMESTAMP WITH TIME ZONE for better timezone handling
        # Using JSONB for efficient JSON storage
        create_table_query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {table} (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            question TEXT NOT NULL,
            rephrased_question TEXT,
            answer TEXT,
            retrieved_contexts JSONB, -- Storing as JSONB
            faithfulness_score REAL,
            response_relevancy_score REAL,
            context_precision_score REAL,
            latency_ms REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER
        );
        """).format(table=sql.Identifier(TABLE_NAME))

        # Execute the SQL statement
        cursor.execute(create_table_query)

        # Commit the changes
        conn.commit()
        print(f"Database '{DB_NAME}' and table '{TABLE_NAME}' are set up successfully in PostgreSQL.")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    create_database_and_table()