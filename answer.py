import sys
import psycopg2
import time
import json
import asyncio

from openai import AsyncOpenAI
from pymilvus import MilvusClient

# 匯入集中化的設定
import config
from prompts import PROMPTS

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

async def retrieve_context(question: str, lang: str = 'zh', top_k: int = 7):
    """根據問題進行相似度檢索+過濾"""
    # 1. 產生問題的向量
    question_embedding = await get_embedding(question)

    # 2. 重問題中提取 metadata 過濾條件
    filters = extract_filters_from_question(question, lang=lang)
    expr = filters_to_expr(filters) if filters else None
    print("Milvus expr:", expr)

    # 3. 執行向量檢索
    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    }

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
        print(f"結果 {i}:")
        print(f"內容: {entity.get('text', '')[:100]}...")
        print(f"相似度: {res.get('distance', 0.0):.4f}, 來源: {entity.get('source_file', 'N/A')}")
        print("-" * 50)

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
        conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
        cursor = conn.cursor()
        
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        total_tokens = usage.total_tokens if usage else None

        insert_query = f"""INSERT INTO {config.DB_TABLE_NAME} 
                         (question, rephrased_question, answer, retrieved_contexts, latency_ms, prompt_tokens, completion_tokens, total_tokens)
                         VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;"""
        
        cursor.execute(insert_query, (question, rephrased_question, answer, json.dumps(contexts, ensure_ascii=False), latency_ms, prompt_tokens, completion_tokens, total_tokens))
        log_id = cursor.fetchone()[0]
        conn.commit()
        print(f"\n[DB] 本次問答紀錄已成功儲存到 PostgreSQL 資料庫，ID: {log_id}。")
        return log_id
    except psycopg2.Error as e:
        print(f"\n[DB Error] 無法寫入 PostgreSQL 資料庫: {e}")
        return None
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

async def _rephrase_question_with_history(history: list, question: str, lang: str = 'zh') -> str:
    """
    使用對話歷史來重構一個新的、獨立的問題。
    """
    if not history:
        return question

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-8:]])
    system_prompt = PROMPTS[lang]['rephrase_system']
    user_prompt = PROMPTS[lang]['rephrase_user'].format(history_str=history_str, question=question)

    try:
        response = await openai_client.chat.completions.create(
            model=config.OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        rephrased_question = response.choices[0].message.content.strip()
        if not rephrased_question:
            return question
        print(f"🔄 重構後的問題: {rephrased_question}")
        return rephrased_question
    except Exception as e:
        print(f"⚠️ 問題重構失敗: {e}")
        return question

async def generate_answer_stream(question: str, cleaned_contexts: list, lang: str = 'zh'):
    """
    把清理過的 Milvus 檢索結果交給 GPT 生成自然語言回答，並以串流形式回傳。
    """
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

    system_prompt = PROMPTS[lang]['rag_system']
    user_prompt = PROMPTS[lang]['rag_user'].format(question=question, context_for_llm=context_for_llm)

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

async def stream_chat_pipeline(question: str, history: list | None = None, lang: str = 'zh'):
    """
    Orchestrates the entire RAG pipeline for streaming responses.
    """
    start_time = time.time()
    full_answer = ""
    original_question = question
    rephrased_question = question
    contexts_for_logging = []
    result_data = {}

    try:
        if history:
            rephrased_question = await _rephrase_question_with_history(history, question, lang=lang)
        
        print(f"\n❓ 最終問題: {rephrased_question} (原始: {original_question})")

        intent = await intent_classification(rephrased_question, lang=lang)
        print(f"意圖: {intent}")

        if intent == "scholarship":
            raw_contexts = await retrieve_context(rephrased_question, lang=lang)
            cleaned_contexts = log_and_clean_contexts(raw_contexts)

            if not cleaned_contexts:
                no_result_answer = PROMPTS[lang]['no_result_answer']
                yield {"type": "content", "data": no_result_answer}
                full_answer = no_result_answer
                result_data = {"contexts": []}
                return

            llm_stream = generate_answer_stream(rephrased_question, cleaned_contexts, lang=lang)
            buffer = ""
            delimiter = "|||SOURCES|||"

            async for chunk in llm_stream:
                full_answer += chunk
                buffer += chunk

                if delimiter in buffer:
                    answer_part, _ = buffer.split(delimiter, 1)
                    yield {"type": "content", "data": answer_part}
                    async for remaining_chunk in llm_stream:
                        full_answer += remaining_chunk
                    break
                else:
                    if len(buffer) > len(delimiter):
                        yield_part = buffer[:-len(delimiter)]
                        yield {"type": "content", "data": yield_part}
                        buffer = buffer[-len(delimiter):]
            else:
                if buffer:
                    yield {"type": "content", "data": buffer}

            answer_part = full_answer
            cited_source_names = []
            if "|||SOURCES|||" in full_answer:
                parts = full_answer.split("|||SOURCES|||")
                answer_part = parts[0].strip()
                source_names_str = parts[1].strip()
                if source_names_str:
                    cited_source_names = [name.strip() for name in source_names_str.split(',')]
            
            full_answer = answer_part

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
            
            result_data = {"contexts": unique_display_contexts}
        
        else: # Small talk
            stream = await openai_client.chat.completions.create(
                model=config.OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": PROMPTS[lang]['small_talk_system']},
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
        
        try:
            log_id = await asyncio.to_thread(log_to_db, original_question, rephrased_question, full_answer, contexts_for_logging, latency_ms, None)
        except Exception as e:
            print(f"[ERROR] log_to_db failed in thread: {e}")
            log_id = None

        if log_id:
            result_data["log_id"] = log_id
        
        yield {"type": "final_data", "data": result_data}
