import os
from openai import OpenAI
from dotenv import load_dotenv
from pymilvus import MilvusClient
from answer import chat_pipeline
load_dotenv()

# --- 初始化 ---
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "rag3_scholarships"

api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=api_key)
milvus_client = MilvusClient(uri=MILVUS_URI)

# 有記憶的聊天
conversation_history = []

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("AI: Goodbye!")
        break

    conversation_history.append({"role": "user", "content": user_input})

    # 嘗試 RAG pipeline
    rag_response = chat_pipeline(user_input)
    if rag_response:
        response = rag_response
    else:
       # 交給 LLM 進行自然對話
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history
        )
        response = completion.choices[0].message.content

    print("=== LLM 回答 ===\n", response)
    conversation_history.append({"role": "assistant", "content": response})
