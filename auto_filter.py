from openai import OpenAI
from dotenv import load_dotenv
from pymilvus import DataType
import json
import os
# 載入 .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

metadata_schema = {
    "status": ["一般生", "原住民", "中低收入戶", "清寒", "低收入戶", "弱勢學生", "境外生", "國際生", "僑生", "港澳生","身心障礙","交換生","畢業生"],
    "subsidy_type": ["海外交流", "獎學金", "獎勵金","助學金","工讀","就學貸款","生活津貼","急難救助","志工服務","社團交流","住宿補助"],
    "edu_system": ["五專", "二技", "專科", "大學部", "碩士班", "博士班"]
}

def extract_filters_from_question(question: str):
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
        model="gpt-4o-mini",
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
    
    # # 轉成 Milvus expr 語法
    # expr_parts = []
    # for k, v in filters.items():
    #     if isinstance(v, list):
    #         values = ",".join([f"'{x}'" for x in v])
    #         expr_parts.append(f"{k} in [{values}]")
    #     else:
    #         expr_parts.append(f"{k} == '{v}'")

    # expr = " and ".join(expr_parts)
    # return expr

def filters_to_expr(filters: dict) -> str:
    """
    將 filter dict 轉換成 Milvus expr 語法
    
    """
    if not filters:
        return ""
    
    expr_parts = []

    for key, value in filters.items():
        if isinstance(value, list) and len(value) > 0 :
            # 多值就當作array - 用 array_contains_any
            values_str = ",".join([f'"{v}"' for v in value])
            expr_parts.append(f'ARRAY_CONTAINS_ANY({key},[{values_str.strip()}])')
        else:
            # expr_parts.append(f'{key} == "{value}"') # 單值
            None

    return " and ".join(expr_parts)




# 測試

if __name__ == "__main__":
    question = "有哪些補助適合低收入戶的大學生？"
    filters = extract_filters_from_question(question)
    print("生成的 metadata 過濾條件:", filters)
    expr = filters_to_expr(filters)
    print("milvus 過濾條件:", expr)
