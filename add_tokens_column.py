
import sqlite3

# --- Constants ---
DB_FILE = "evaluation.db"
TABLE_NAME = "qa_logs"

def add_token_usage_columns():
    """
    Adds columns for token usage to the qa_logs table.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Define the columns to add
        columns_to_add = [
            ("prompt_tokens", "INTEGER"),
            ("completion_tokens", "INTEGER"),
            ("total_tokens", "INTEGER")
        ]

        # Check existing columns to avoid errors on re-run
        cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
        existing_columns = [info[1] for info in cursor.fetchall()]

        for column_name, column_type in columns_to_add:
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {column_name} {column_type}")
                print(f"Column '{column_name}' added to table '{TABLE_NAME}'.")
            else:
                print(f"Column '{column_name}' already exists in table '{TABLE_NAME}'.")

        conn.commit()
        print("\nDatabase schema updated successfully.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    add_token_usage_columns()
