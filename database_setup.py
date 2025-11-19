import os
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the table name as a constant
TABLE_NAME = "qa_logs"

def create_database_and_table():
    """
    Connects to the PostgreSQL database and creates the qa_logs table
    if it hasn't been created yet.
    """
    try:
        # 1. 在函式內部讀取環境變數
        conn_params = {
            'host': os.getenv("DB_HOST"),
            'port': os.getenv("DB_PORT", "5432"), # 提供預設 port
            'dbname': os.getenv("DB_NAME"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD")
        }
        
        # 檢查是否有遺漏的參數
        for key, value in conn_params.items():
            if not value:
                raise ValueError(f"Missing database connection parameter: {key}")

        print(f"嘗試連線到資料庫: {conn_params['host']}/{conn_params['dbname']}")

        # 2. 使用 'with' 語句自動管理連線和遊標
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cursor:
                create_table_query = sql.SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    question TEXT NOT NULL,
                    rephrased_question TEXT,
                    answer TEXT,
                    retrieved_contexts JSONB,
                    faithfulness_score REAL,
                    response_relevancy_score REAL,
                    context_precision_score REAL,
                    latency_ms REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER
                );
                """).format(table=sql.Identifier(TABLE_NAME))

                cursor.execute(create_table_query)
                # conn.commit() 在 with conn 區塊結束時自動調用
                print(f"✅ Database '{conn_params['dbname']}' and table '{TABLE_NAME}' are set up successfully in PostgreSQL.")

    except ValueError as ve:
        # 專門捕獲我們自己拋出的環境變數錯誤
        print(f"❌ Configuration Error: {ve}")
    except psycopg2.Error as e:
        # 捕獲所有資料庫相關錯誤
        print(f"❌ Database error: {e}")
    except Exception as e:
        # 捕獲其他所有意外錯誤
        print(f"❌ An unexpected error occurred: {e}")

# 確保腳本可以獨立運行
if __name__ == "__main__":
    create_database_and_table()