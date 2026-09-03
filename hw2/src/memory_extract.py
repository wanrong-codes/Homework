
import json
import os
import re
import uuid

import ollama

EXTRACT_PROMPT = """你是一个记忆抽取助手。请阅读下面这段用户与AI助手的聊天记录，
从中抽取三类信息，只输出JSON，不要输出任何其他文字、不要用markdown代码块包裹。

三类信息定义:
- preferences: 用户表达出的偏好/喜好(比如喜欢的风格、工具、口味等)
- facts: 用户明确陈述的、关于自己的客观事实(比如职业、所在城市、养的宠物等)
- patterns: 从多轮对话中观察到的重复出现的行为模式(比如经常在某个时间段问某类问题)

输出格式严格如下(数组内每一项都是一句简短的中文描述):
{{
  "preferences": ["...", "..."],
  "facts": ["...", "..."],
  "patterns": ["...", "..."]
}}

如果某一类没有可抽取的内容，返回空数组，不要编造。

聊天记录:
---
{chat_history}
---
"""


def extract_memory(chat_history: str, model: str = "qwen2.5:7b") -> dict:
    prompt = EXTRACT_PROMPT.format(chat_history=chat_history)
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    raw = resp["message"]["content"].strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM未返回合法JSON，原始输出:\n{raw}") from e
    for key in ("preferences", "facts", "patterns"):
        data.setdefault(key, [])
    return data


def store_memory(extracted: dict, embedder, memory_store):
    ids, texts, metadatas = [], [], []
    for category in ("preferences", "facts", "patterns"):
        for item in extracted.get(category, []):
            ids.append(str(uuid.uuid4()))
            texts.append(item)
            metadatas.append({"category": category})

    if not texts:
        return 0

    vecs = embedder.embed_documents(texts)
    memory_store.add(ids=ids, embeddings=vecs, documents=texts, metadatas=metadatas)
    return len(texts)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from embedding import OllamaEmbedder
    from vector_store import ChromaStore

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chat_path = os.path.join(here, "experiments", "sample_chat_history.txt")
    with open(chat_path, "r", encoding="utf-8") as f:
        chat_history = f.read()

    extracted = extract_memory(chat_history)
    print("抽取结果:")
    print(json.dumps(extracted, ensure_ascii=False, indent=2))

    embedder = OllamaEmbedder() 
    memory_store = ChromaStore(persist_dir=os.path.join(here, "memory_db"),
                                collection_name="user_memory")
    n = store_memory(extracted, embedder, memory_store)
    print(f"\n已存入 {n} 条memory到独立的memory_db")
