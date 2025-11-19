
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_table():
    """Connects to the PostgreSQL database and creates the qa_logs table if it doesn't exist."""
    conn = None
    cursor = None
    try:
        # Connect to the database using environment variables
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("❌ DATABASE_URL environment variable not set.")
        cursor = conn.cursor()

        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()


        TABLE_NAME = "qa_logs"

        # SQL statement to create the table
        # Using "IF NOT EXISTS" makes the script safe to run multiple times.
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            question TEXT,
            rephrased_question TEXT,
            answer TEXT,
            retrieved_contexts JSONB,
            latency_ms REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """

        cursor.execute(create_table_query)
        conn.commit()
        print(f"✅ Table '{TABLE_NAME}' created successfully or already exists.")

    except psycopg2.Error as e:
        print(f"❌ [DB Connection/Query Error] Could not create table: {e}")
    except Exception as e:
        print(f"❌ [Unexpected Error] An error occurred during database setup: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    print("🚀 Running database setup script...")
    create_table()
    print("🏁 Database setup script finished.")
