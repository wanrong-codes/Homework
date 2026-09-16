
```bash
cd hw3
python3 -m venv .venv && source .venv/bin/activate   # 可选
pip install -r requirements.txt                       # Mock 模式其实不需要装 openai，也能跑

python demo_function_calling.py
python demo_react.py
python demo_skills_and_memory.py

python -m pytest tests/ -v      # 或者 python tests/test_agents.py
```


## 切换到真实 LLM

复制 `.env.example` 为 `.env`（或者直接 `export`），设置：

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-xxxx
export OPENAI_MODEL=gpt-4o-mini
```


```bash
export LLM_PROVIDER=openai
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export OPENAI_MODEL=llama3.1
```

