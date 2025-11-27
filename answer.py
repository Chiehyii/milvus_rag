import sys
import psycopg2
import time
import json
import asyncio

from openai import AsyncOpenAI
from pymilvus import MilvusClient

# 匯入集中化的設定
import config

from auto_filter import extract_filters_from_question, filters_to_expr
from intent_classification import intent_classification

# 使用集中化的設定來初始化 clients
openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
milvus_client = MilvusClient(
    uri=config.CLUSTER_ENDPOINT,
    token=config.ZILLIZ_API_KEY,
)

async def get_embedding(text):
    """產生文字向量"""
    resp = await openai_client.embeddings.create(
        input=text,
        model=config.EMBEDDING_MODEL
    )
    return resp.data[0].embedding

async def retrieve_context(question: str, top_k: int=7):
    """根據問題進行相似度檢索+過濾"""
    # 1. 產生問題的向量
    question_embedding = await get_embedding(question)

    # 2. 重問題中提取 metadata 過濾條件
    filters = extract_filters_from_question(question)
    expr = filters_to_expr(filters) if filters else None
    print("Milvus expr:", expr)

    # 3. 執行向量檢索
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10} # 確保這裡沒有 "expr"
    }

    # milvus_client.search is synchronous; run in a thread to avoid blocking the event loop
    def _milvus_search():
        return milvus_client.search(
            collection_name=config.MILVUS_COLLECTION,
            data=[question_embedding],
            search_params=search_params,
            limit=top_k,
            filter=expr if expr else None,
            output_fields=["id", "text", "source_file", "source_url", "status", "subsidy_type", "edu_system"],
        )

    results = await asyncio.to_thread(_milvus_search)
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


def log_to_db(question, rephrased_question, answer, contexts, latency_ms, usage):
    """將問答資料和 token 使用量記錄到 PostgreSQL 資料庫中"""
    conn = None
    cursor = None
    try:
        # # 使用 config 裡的連線資訊
        conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
        # db_url = os.getenv("DATABASE_URL")
        # if not db_url:
        #     raise ValueError("\n[DB Error]❌ DATABASE_ environment variable not set.")
        
        # conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # 確保 retrieved_contexts 是合法的 JSON 字串
        # psycopg2 會自動處理 Python dict 到 JSONB 的轉換
        
        # 從 usage 物件中安全地獲取 token 資訊
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        total_tokens = usage.total_tokens if usage else None

        insert_query = f"""INSERT INTO {config.DB_TABLE_NAME} 
                         (question, rephrased_question, answer, retrieved_contexts, latency_ms, prompt_tokens, completion_tokens, total_tokens)
                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;"""
        
        # 將 contexts python dict 直接傳遞，psycopg2 會將其序列化為 JSON
        cursor.execute(insert_query, (question, rephrased_question, answer, json.dumps(contexts, ensure_ascii=False), latency_ms, prompt_tokens, completion_tokens, total_tokens))
        
        # 獲取返回的 id
        log_id = cursor.fetchone()[0]
        
        conn.commit()
        print(f"\n[DB] 本次問答紀錄已成功儲存到 PostgreSQL 資料庫，ID: {log_id}。")
        return log_id

    except psycopg2.Error as e:
        print(f"\n[DB Error] 無法寫入 PostgreSQL 資料庫: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

async def _rephrase_question_with_history(history: list, question: str) -> str:
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
        response = await openai_client.chat.completions.create(
            model=config.OPENAI_MODEL_NAME,
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


# --- Streaming Functions ---
# ------------------------------------------------- 生成答案--------------------------------------------------

async def generate_answer_stream(question: str, cleaned_contexts: list):
    """
    把清理過的 Milvus 檢索結果交給 GPT 生成自然語言回答，並以串流形式回傳。
    這是一個生成器函式。
    """
    # (The logic for preparing context_for_llm is identical to the non-streaming version)
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

    stream = await openai_client.chat.completions.create(
        model=config.OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        stream=True,
    )
    async for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        yield content

async def stream_chat_pipeline(question: str, history: list | None = None):
    """
    Orchestrates the entire RAG pipeline for streaming responses.
    This is an async generator that yields different types of events.
    """
    start_time = time.time()
    full_answer = ""
    original_question = question
    rephrased_question = question
    contexts_for_logging = []
    result_data = {} # Define here to ensure it's available in finally

    try:
        if history:
            rephrased_question = await _rephrase_question_with_history(history, question)
        
        print(f"\n❓ 最終問題: {rephrased_question} (原始: {original_question})")

        intent = await intent_classification(rephrased_question)
        print(f"意圖: {intent}")

        if intent == "scholarship":
            raw_contexts = await retrieve_context(rephrased_question)
            cleaned_contexts = log_and_clean_contexts(raw_contexts)

            if not cleaned_contexts:
                no_result_answer = "抱歉，我沒有找到相關的補助或獎學金資訊。"
                yield {"type": "content", "data": no_result_answer}
                full_answer = no_result_answer
                result_data = {"contexts": []} # Set for finally block
                return

            # Stream the answer from LLM, but filter out the source part
            llm_stream = generate_answer_stream(rephrased_question, cleaned_contexts)
            buffer = ""
            delimiter = "|||SOURCES|||"

            # This loop yields the answer part of the stream and stops before the sources.
            # It still accumulates the `full_answer` in the background for parsing later.
            async for chunk in llm_stream:
                full_answer += chunk
                buffer += chunk

                if delimiter in buffer:
                    # Delimiter found. Yield the part before it and stop yielding content.
                    answer_part, _ = buffer.split(delimiter, 1)
                    yield {"type": "content", "data": answer_part}

                    # Consume the rest of the stream without yielding content
                    async for remaining_chunk in llm_stream:
                        full_answer += remaining_chunk
                    break
                else:
                    # To keep the stream flowing, yield parts of the buffer that we know
                    # don't contain the full delimiter. We hold back a small part.
                    if len(buffer) > len(delimiter):
                        yield_part = buffer[:-len(delimiter)]
                        yield {"type": "content", "data": yield_part}
                        buffer = buffer[-len(delimiter):]
            else:
                # If the loop finishes without finding the delimiter, yield any remaining buffer content.
                if buffer:
                    yield {"type": "content", "data": buffer}

            # Parse the full answer to get cited sources
            answer_part = full_answer
            cited_source_names = []
            if "|||SOURCES|||" in full_answer:
                parts = full_answer.split("|||SOURCES|||")
                answer_part = parts[0].strip()
                source_names_str = parts[1].strip()
                if source_names_str:
                    cited_source_names = [name.strip() for name in source_names_str.split(',')]
            
            full_answer = answer_part # Update full_answer to be only the user-facing part

            cited_source_names_set = set(cited_source_names)
            all_cited_contexts = [ctx for ctx in cleaned_contexts if ctx.get('source_file') in cited_source_names_set]
            contexts_for_logging = all_cited_contexts

            unique_display_contexts = []
            seen_keys = set()
            for context in all_cited_contexts:
                unique_key = context.get('source_url') or context.get('source_file')
                if unique_key not in seen_keys:
                    unique_display_contexts.append(context)
                    seen_keys.add(unique_key)
            
            # This is the final payload
            result_data = {"contexts": unique_display_contexts}
        
        else: # Small talk
            stream = await openai_client.chat.completions.create(
                model=config.OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一個慈濟大學的聊天助理，主要提供獎助學金和補助資訊。請自然且簡短地回應，並引導使用者提問相關問題。若問題無關，請禮貌地表示無法回答。"},
                    {"role": "user", "content": rephrased_question}
                ],
                temperature=0.7,
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                full_answer += content
                yield {"type": "content", "data": content}
            
            result_data = {"contexts": []}

    finally:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        print(f"\n⏱️ 本次問答總耗時: {latency_ms:.2f} ms")
        
        # Log to DB (usage is None for streaming)
        # log_to_db is synchronous; run it in a thread to avoid blocking
        try:
            log_id = await asyncio.to_thread(log_to_db, original_question, rephrased_question, full_answer, contexts_for_logging, latency_ms, None)
        except Exception as e:
            print(f"[ERROR] log_to_db failed in thread: {e}")
            log_id = None

        if log_id:
            result_data["log_id"] = log_id
        
        # Yield the final data packet
        yield {"type": "final_data", "data": result_data}
