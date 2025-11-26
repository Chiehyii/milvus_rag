import sys
import asyncio
sys.path.append('d:/git_sourcetree/milvus_rag')

import answer

# Monkeypatch intent_classification to always return 'other'
async def fake_intent(question: str) -> str:
    await asyncio.sleep(0)
    return 'other'

answer.intent_classification = fake_intent

# Monkeypatch openai_client.chat.completions.create to return an async iterator of chunks
async def fake_create_stream(*args, **kwargs):
    async def gen():
        # Simulate streaming chunks
        chunks = ["Hello", " world!", " This is a test."]
        for c in chunks:
            # Create an object with structure chunk.choices[0].delta.content
            class Delta:
                def __init__(self, content):
                    self.content = content

            class Choice:
                def __init__(self, delta):
                    self.delta = delta

            class Chunk:
                def __init__(self, choice):
                    self.choices = [choice]

            yield Chunk(Choice(Delta(c)))
            await asyncio.sleep(0)
    return gen()

# Attach to openai_client
if not hasattr(answer, 'openai_client'):
    class Dummy:
        pass
    answer.openai_client = Dummy()

if not hasattr(answer.openai_client, 'chat'):
    class _Chat:
        pass
    answer.openai_client.chat = _Chat()

if not hasattr(answer.openai_client.chat, 'completions'):
    class _Comp:
        pass
    answer.openai_client.chat.completions = _Comp()

answer.openai_client.chat.completions.create = fake_create_stream

async def run_test():
    print('Starting smoke test for stream_chat_pipeline')
    async for event in answer.stream_chat_pipeline('測試問題', []):
        print('EVENT:', event)
    print('Smoke test finished')

if __name__ == '__main__':
    asyncio.run(run_test())
