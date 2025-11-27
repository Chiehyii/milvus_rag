import os
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Zilliz / Milvus ---
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")
CLUSTER_ENDPOINT = os.getenv("CLUSTER_ENDPOINT")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag5_scholarships_hybrid")

# --- PostgreSQL Database ---
DB_TABLE_NAME = os.getenv("DB_TABLE_NAME", "qa_logs2")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# --- CORS ---
# 從環境變數讀取允許的來源，預設為本地開發常用的來源
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,https://tcu-scholarships-chatbot.onrender.com/")
# 將字串轉換為列表
ALLOWED_ORIGINS_LIST = [origin.strip() for origin in CORS_ALLOWED_ORIGINS.split(',')]

# 簡單檢查以確保關鍵環境變數已設定
if not OPENAI_API_KEY or not ZILLIZ_API_KEY or not CLUSTER_ENDPOINT:
    raise ValueError("遺失關鍵環境變數： OPENAI_API_KEY, ZILLIZ_API_KEY, 或 CLUSTER_ENDPOINT 必須被設定。")
