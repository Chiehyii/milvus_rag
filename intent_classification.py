import config
from openai import AsyncOpenAI

# 建立 OpenAI client
client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

INTENT_DEFINITIONS = {
    "scholarship": "問題與獎助學金、補助、助學金、生活津貼、工讀等相關詳情或根據資格推薦獎學金",
    "other": "打招呼、寒暄或閒聊、其他問題"
}

async def intent_classification(question: str) -> str:
    """
    輸入問題，輸出意圖分類。
    意圖類別從 INTENT_DEFINITIONS 動態生成。
    """
    
    # 從 INTENT_DEFINITIONS 動態生成提示選項
    intent_options = "\n".join([f"{i+1}. {name} → {desc}" for i, (name, desc) in enumerate(INTENT_DEFINITIONS.items())])
    
    prompt = f"""
    請將以下問題分類為其中之一：
{intent_options}

    問題: {question}

    只輸出類別名稱（例如 "scholarship" 或 "other"），不要多餘的文字。
    """

    resp = await client.chat.completions.create(
        model=config.OPENAI_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    intent = resp.choices[0].message.content.strip().lower()

    # 從 INTENT_DEFINITIONS 的鍵來做安全檢查，如果都不是，就預設為 "other"
    if intent not in INTENT_DEFINITIONS:
        intent = "other"

    return intent