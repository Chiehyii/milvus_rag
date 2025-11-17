from openai import OpenAI

client = OpenAI()

def intent_classification(question: str) -> str:
    """
    輸入問題，輸出意圖分類：
    - scholarship: 與獎助學金/補助相關
    - other: 打招呼、閒聊、其他問題
    """
    prompt = f"""
    請將以下問題分類為三種之一：
    1. scholarship → 問題與獎學金、補助、助學金、生活津貼、工讀等相關詳情或根據資格推薦獎學金
    2. other → 打招呼、寒暄或閒聊、其他問題

    問題: {question}

    只輸出類別，不要多餘的文字。
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    intent = resp.choices[0].message.content.strip().lower()

    # 安全檢查
    if intent not in ["scholarship", "other"]:
        intent = "other"

    return intent