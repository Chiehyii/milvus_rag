from openai import OpenAI
from pymilvus import DataType
import json
import config
from prompts import PROMPTS

client = OpenAI(api_key=config.OPENAI_API_KEY)

def extract_filters_from_question(question: str, lang: str = 'zh', schema_path: str = "metadata_schema.json"):
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

    prompt = PROMPTS[lang]['filter_extraction_system'].format(
        metadata_schema=metadata_schema,
        question=question
    )

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
    question_zh = "有哪些補助適合低收入戶的大學生？"
    filters_zh = extract_filters_from_question(question_zh, lang='zh')
    print("生成的 metadata 過濾條件 (zh):", filters_zh)
    expr_zh = filters_to_expr(filters_zh)
    print("milvus 過濾條件 (zh):", expr_zh)

    question_en = "What subsidies are available for low-income university students?"
    filters_en = extract_filters_from_question(question_en, lang='en')
    print("\nGenerated metadata filters (en):", filters_en)
    expr_en = filters_to_expr(filters_en)
    print("milvus filter expression (en):", expr_en)