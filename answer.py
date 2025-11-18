import os
import psycopg2
import time
import json
from dotenv import load_dotenv

from openai import OpenAI
from pymilvus import MilvusClient
from intent_classification import intent_classification

# Load environment variables from .env file
load_dotenv()

zilliz_api_key = os.getenv("ZILLIZ_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=api_key)
CLUSTER_ENDPOINT="https://in03-a6f08ce2ff778ed.serverless.gcp-us-west1.cloud.zilliz.com:443"
milvus_client = MilvusClient(
                    uri=CLUSTER_ENDPOINT,
                    token=zilliz_api_key,
                    )
collection_name = "rag5_scholarships_hybrid"

def get_embedding(text):
    """產生文字向量"""
    resp = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return resp.data[0].embedding

def retrieve_context(question: str, top_k: int=7):
    """根據問題進行相似度檢索+過濾"""
    # 1. 產生問題的向量
    question_embedding = get_embedding(question)

    # 2. 重問題中提取 metadata 過濾條件
    # filters = extract_filters_from_question(question)
    # expr = filters_to_expr(filters) if filters else None
    # print("Milvus expr:", expr)

    # 3. 執行向量檢索
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10} # 確保這裡沒有 "expr"
    }

    results = milvus_client.search(
        collection_name=collection_name,
        data=[question_embedding],
        search_params=search_params,
        limit=top_k,
        # filter=expr if expr else None,
        output_fields=["id", "text", "source_file", "source_url", "status", "subsidy_type", "edu_system"],
    )
    if not results or not results[0]:
        return []

    return results[0]

def log_and_clean_contexts(retrieved_docs: list):
    """
    將檢索結果打印到控制台，並返回一個清理過的、可序列化的列表。
    """
    print("\n=== RAG檢索結果 ===")
    if not retrieved_docs:
        print("（沒有檢索到任何結果）")
        return []

    cleaned_contexts = []
    for i, res in enumerate(retrieved_docs, 1):
        entity = res.get("entity", {})

        # 打印日誌
        print(f"結果 {i}:")
        print(f"內容: {entity.get('text', '')[:100]}...")
        print(f"相似度: {res.get('distance', 0.0):.4f}, 來源: {entity.get('source_file', 'N/A')}")
        print("-" * 50)

        # 準備清理過的資料
        # 從 Milvus 獲取的 ARRAY 類型可能是無法直接序列化的 Protobuf 類型/容器,
        # 在此處手動轉換為 python list
        status = entity.get("status")
        subsidy_type = entity.get("subsidy_type")
        edu_system = entity.get("edu_system")

        cleaned_contexts.append({
            "id": res.get("id"),
            "text": entity.get("text"),
            "source_file": entity.get("source_file", "").replace(".md", ""),
            "source_url": entity.get("source_url"),
            "status": list(status) if status else [],
            "subsidy_type": list(subsidy_type) if subsidy_type else [],
            "edu_system": list(edu_system) if edu_system else [],
            "distance": res.get("distance")
        })

    return cleaned_contexts

# ------------------------------------------------- 生成答案--------------------------------------------------
def generate_answer(question: str, cleaned_contexts: list):
    """把清理過的 Milvus 檢索結果交給 GPT 生成自然語言回答，並返回完整的 API 回應"""

    # ... (The existing logic for preparing context_for_llm remains the same)
    from collections import defaultdict
    grouped = defaultdict(list)
    source_url_map = {}
    for c in cleaned_contexts:
        fname = c.get('source_file', '未知來源')
        grouped[fname].append(c.get('text', ''))
        if fname not in source_url_map and c.get('source_url'):
            source_url_map[fname] = c.get('source_url')

    context_for_llm = ""
    for fname, texts in grouped.items():
        title = fname.replace('.md', '').replace('.txt', '')
        url = source_url_map.get(fname, '')
        context_for_llm += f"\n---\n來源名稱: {title}\n"
        if url:
            context_for_llm += f"來源網址: {url}\n"
        full_text = "\n".join(texts)
        context_for_llm += f"內容: {full_text}\n"

    system_prompt = f"""你是一個專業的慈濟大學獎學金問答助理。你的任務是根據提供的「檢索內容」來回答「使用者問題」。

    **輸出格式**
    你的輸出必須嚴格包含兩部分，並由一個特殊的分隔符號 `|||SOURCES|||` 隔開。

    **第一部分：給使用者的回答**
    1.  **分析**：仔細分析「檢索內容」，判斷哪些來源與「使用者問題」真正相關。
    2.  **生成回答**：
        * 如果有多個獎助學金種類就為每個獎學金或補助建立一個獨立的段落。
        * 只回答和「使用者問題」直接相關的資訊，避免包含不相關的細節。
        * 如果「檢索內容」中沒有任何資訊能回答「使用者問題」，請禮貌地告知使用者你無法回答，而不是編造資訊。
        * 每個段落都必須以分點列出，並必須遵循以下格式獨立呈現：
            * 標題：該獎學金的「來源名稱」作為標題（使用 Markdown 的 `**粗體**` 格式）。
            * 內容：根據檢索內容中，以流暢的段落或項目符號來呈現。
        * 在標題下方，僅使用相關的內容來組織你的回答。
        * 使用自然的語言和 Markdown 排版（粗體、項目符號等）來美化輸出。
    3.  **禁止**：不要在這部分包含任何關於資料來源的文字（標題除外）。

    **第二部分：資料來源列表**
    1.  在分隔符號 `|||SOURCES|||` 之後，你必須列出你在第一部分回答中，所使用到的所有「來源名稱」。
    2.  格式為一個簡單的、由逗號分隔的字串，例如：`來源名稱一,來源名稱二`。
    3.  如果根據「檢索內容」無法回答問題，則這部分應為空。

    """

    user_prompt = f"""
    使用者問題：
    {question}

    檢索內容：
    {context_for_llm}
    """

    # 返回完整的 response 物件
    return openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
    )

def log_to_db(question, rephrased_question, answer, contexts, latency_ms, usage):
    """將問答資料和 token 使用量記錄到 PostgreSQL 資料庫中"""
    conn = None
    cursor = None
    try:
        # 從環境變數讀取 PostgreSQL 連線資訊
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )
        cursor = conn.cursor()
        
        # 確保 retrieved_contexts 是合法的 JSON 字串
        # psycopg2 會自動處理 Python dict 到 JSONB 的轉換
        
        # 從 usage 物件中安全地獲取 token 資訊
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        total_tokens = usage.total_tokens if usage else None

        TABLE_NAME = "qa_logs"
        insert_query = f"""INSERT INTO {TABLE_NAME} 
                         (question, rephrased_question, answer, retrieved_contexts, latency_ms, prompt_tokens, completion_tokens, total_tokens)
                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
        
        # 將 contexts python dict 直接傳遞，psycopg2 會將其序列化為 JSON
        cursor.execute(insert_query, (question, rephrased_question, answer, json.dumps(contexts, ensure_ascii=False), latency_ms, prompt_tokens, completion_tokens, total_tokens))
        conn.commit()
        print("\n[DB] 本次問答紀錄已成功儲存到 PostgreSQL 資料庫。")

    except psycopg2.Error as e:
        print(f"\n[DB Error] 無法寫入 PostgreSQL 資料庫: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def _rephrase_question_with_history(history: list, question: str) -> str:
    """
    使用對話歷史來重構一個新的、獨立的問題。
    """
    if not history:
        return question

    # 將 history 轉換為適合 LLM 的格式
    # 為避免過長，只取最近的 4 輪對話
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-8:]])

    system_prompt = """你是一個對話助理，你的任務是根據提供的「對話歷史」和「最新的使用者問題」，生成一個獨立、完整的「重構後的問題」。
這個「重構後的問題」必須能夠在沒有任何上下文的情況下被完全理解。

**規則:**
- 如果「最新的使用者問題」**不是一個問題** (例如：道謝 "謝謝", 肯定 "我知道了", 問候 "你好"), **請直接原樣返回「最新的使用者問題」**，不要做任何改寫。
- 如果「最新的使用者問題」已經是一個完整的、可獨立理解的問題，直接返回原問題。
- 否則，請結合「對話歷史」來改寫問題，使其變得完整。
- 保持問題簡潔。

例如 (需要改寫):
對話歷史:
user: 我想找清寒獎學金
assistant: 我們有幾種清寒獎學金，例如 A 和 B。
最新的使用者問題:
它需要什麼資格?

重構後的問題:
申請 B 清寒獎學金需要什麼資格？

例如 (無需改寫):
對話歷史:
user: 慈濟醫療法人獎助學金的申請流程是什麼？
assistant: 申請流程是...
最新的使用者問題:
謝謝

重構後的問題:
謝謝
"""

    user_prompt = f"""對話歷史:
{history_str}

最新的使用者問題:
{question}

重構後的問題:
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=150, # 限制輸出長度
        )
        rephrased_question = response.choices[0].message.content.strip()
        # 避免返回空字串
        if not rephrased_question:
            return question
        print(f"🔄 重構後的問題: {rephrased_question}")
        return rephrased_question
    except Exception as e:
        print(f"⚠️ 問題重構失敗: {e}")
        return question # Fallback to the original question

def chat_pipeline(question: str, history: list | None = None):
    start_time = time.time()
    result = {}
    usage = None
    original_question = question # 保存原始問題以供日誌記錄
    contexts_for_logging = [] # 用於儲存完整的、未經過去重的上下文，以便日誌記錄

    try:
        # 如果有歷史紀錄，重構問題
        if history:
            question = _rephrase_question_with_history(history, question)
        
        print(f"\n❓ 最終問題: {question} (原始: {original_question})")

        intent = intent_classification(question)
        print(f"意圖: {intent}")

        if intent == "scholarship":
            raw_contexts = retrieve_context(question)
            cleaned_contexts = log_and_clean_contexts(raw_contexts)

            if not cleaned_contexts:
                result = {"answer": "抱歉，我沒有找到相關的補助或獎學金資訊。","contexts":[]}
                contexts_for_logging = [] # 確保在返回前賦值
                return result
            
            # Step 4: 獲取完整的 API 回應
            llm_response = generate_answer(question, cleaned_contexts)
            llm_output = llm_response.choices[0].message.content.strip()
            usage = llm_response.usage # 保存 usage 物件

            # Step 5: 解析 LLM 輸出
            answer = llm_output
            cited_source_names = []
            if "|||SOURCES|||" in llm_output:
                parts = llm_output.split("|||SOURCES|||")
                answer = parts[0].strip()
                source_names_str = parts[1].strip()
                if source_names_str:
                    cited_source_names = [name.strip() for name in source_names_str.split(',')]

            # Step 6: 過濾出完整的引用上下文，用於日誌記錄
            cited_source_names_set = set(cited_source_names)
            all_cited_contexts = []
            for context in cleaned_contexts:
                if context.get('source_file') in cited_source_names_set:
                    all_cited_contexts.append(context)
            
            contexts_for_logging = all_cited_contexts # 將完整列表賦值給日誌專用變數

            # Step 7: 建立一個去重的版本，用於前端顯示
            unique_display_contexts = []
            seen_keys = set()
            for context in all_cited_contexts:
                # 優先使用 URL 作為唯一標識，若無則使用檔名
                unique_key = context.get('source_url') or context.get('source_file')
                if unique_key not in seen_keys:
                    unique_display_contexts.append(context)
                    seen_keys.add(unique_key)
            
            result = {"answer": answer, "contexts": unique_display_contexts} # 回傳給前端的是去重後的版本

            print(f"💡 LLM 回答: {result['answer']}")
            if result["contexts"]:
                print("\n--- LLM 實際參考來源 ---")
                for i, context in enumerate(result["contexts"], 1):
                    print(f"{i}. {context.get('source_file', 'N/A')}")
                print("-------------------------")

            return result

        else:
            # 對於閒聊，同樣獲取完整回應
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一個慈濟大學的聊天助理，主要提供獎助學金和補助資訊。請自然且簡短地回應，並引導使用者提問相關問題。若問題無關，請禮貌地表示無法回答。"},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
            usage = resp.usage # 保存 usage 物件
            result = {"answer": resp.choices[0].message.content.strip(), "contexts": []}
            contexts_for_logging = [] # 確保在返回前賦值
            print(f"💡 LLM 回答: {result['answer']}")
            return result
    finally:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        print(f"\n⏱️ 本次問答總耗時: {latency_ms:.2f} ms")
        
        final_answer = result.get("answer", "")
        
        # 使用專門為日誌準備的、未經過去重的完整上下文列表
        log_to_db(original_question, question, final_answer, contexts_for_logging, latency_ms, usage)
