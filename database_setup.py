
import sqlite3

# --- Constants ---
DB_FILE = "evaluation.db"
TABLE_NAME = "qa_logs"

def create_database_and_table():
    """
    Connects to the SQLite database (creating it if it doesn't exist)
    and creates the qa_logs table if it hasn't been created yet.
    """
    try:
        # Connect to SQLite database. This will create the file if it does not exist.
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # SQL statement to create a table
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            question TEXT NOT NULL,
            rephrased_question TEXT,
            answer TEXT,
            retrieved_contexts TEXT, -- Storing as JSON string
            faithfulness_score REAL,
            response_relevancy_score REAL,
            context_precision_score REAL,
            latency_ms REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER
        );
        """

        # Execute the SQL statement
        cursor.execute(create_table_query)

        # Commit the changes and close the connection
        conn.commit()
        print(f"Database '{DB_FILE}' and table '{TABLE_NAME}' are set up successfully.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_database_and_table()
