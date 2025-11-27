from openai import OpenAI
from pymilvus import DataType
import json
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

def extract_filters_from_question(question: str, schema_path: str = "metadata_schema.json"):
    # 從文件加載 metadata schema
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            metadata_schema = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Schema 檔案 '{schema_path}' 不存在。")
        return {}
    except json.JSONDecodeError:
        print(f"⚠️ 無法解析 Schema 檔案 '{schema_path}'。")
        return {}

    prompt = f"""
    你是一個的檢索條件生成器。
    你的任務是根據提供的 metadata schema，從問題中找出對應的欄位與值。

    Schema: {metadata_schema}

    輸出要求：
    1. 只選擇 schema 裡最相似的詞，不要自己創造新值。
    2. 如果找不到對應的值，就不要輸出該欄位，不要猜測或擴展。
    3. 僅輸出純 JSON，不能有多餘的文字, 不要有json 標記。
    4. 不要輸出空值或空陣列。
    5. 即使只有一個值，也請用陣列形式輸出，例如 "status": ["一般生"]。

    現在，請根據以上規則處理一下問題：
    問題: {question}

    """

    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0  # 降低隨機性
    )

    raw_text = resp.choices[0].message.content.strip()
    print("🔎 原始 LLM 輸出:", raw_text)  # 方便 debug

    try:
        filters = json.loads(raw_text)
        return filters
    except Exception as e:
        print("⚠️ JSON parse 失敗:", e)
        return {}


def filters_to_expr(filters: dict) -> str:
    """
    將 filter dict 轉換成 Milvus expr 語法
    """
    if not filters:
        return ""
    
    expr_parts = []
    for key, value in filters.items():
        # LLM 被指示要回傳陣列，所以我們只處理 value 是非空陣列的格式
        if isinstance(value, list) and value:
            # 使用 ARRAY_CONTAINS_ANY 檢查 JSON 欄位是否包含任何指定的值
            values_str = ", ".join([f'"{v}"' for v in value])
            expr_parts.append(f'ARRAY_CONTAINS_ANY({key}, [{values_str}])')

    return " and ".join(expr_parts)

# 測試

if __name__ == "__main__":
    question = "有哪些補助適合低收入戶的大學生？"
    filters = extract_filters_from_question(question)
    print("生成的 metadata 過濾條件:", filters)
    expr = filters_to_expr(filters)
    print("milvus 過濾條件:", expr)