from locust import HttpUser, task, between
import random

class ChatbotUser(HttpUser):
    """
    模擬與 Chatbot API 互動的用戶。
    """
    # 每個任務執行之間的等待時間(秒)，模擬用戶思考。
    wait_time = between(1, 5)

    # 要測試的 API 的 host。
    # 這應該與您的 FastAPI 應用程式運行的地址相符。
    host = "http://127.0.0.1:8000"

    def on_start(self):
        """
        當一個 Locust 模擬用戶啟動時調用。
        在這裡定義一組範例問題。
        """
        self.sample_questions = [
            "有甚麼獎學金推薦?",
            "低收入戶可以申請什麼補助?",
            "我是原住民學生，請問有什麼獎學金嗎?",
            "你好",
            "慈濟大學在哪裡",
            "我想申請海外交流補助"
        ]

    @task
    def chat(self):
        """
        每個模擬用戶將執行的主要任務。
        它會發送一個帶有隨機問題的 POST 請求到 /chat 端點。
        """
        # 從列表中隨機選擇一個問題
        random_question = random.choice(self.sample_questions)

        # 定義 POST 請求的 JSON 內容
        payload = {
            "query": random_question
        }

        # 發送 POST 請求到 /chat 端點
        self.client.post(
            "/chat",
            json=payload,
            name="/chat"  # 在統計數據中將所有聊天請求歸為一類
        )

# 如何運行測試:
# 1. 確認您的 FastAPI 伺服器正在運行: uvicorn main:app --reload
# 2. 開一個新的終端機，然後運行 Locust: locust -f locustfile.py
# 3. 打開您的瀏覽器，訪問 http://localhost:8089
# 4. 輸入要模擬的用戶總數和每秒產生的用戶數，然後開始測試。
