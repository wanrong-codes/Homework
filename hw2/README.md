

```bash
pip install -r requirements.txt

ollama pull qwen2.5:7b
ollama pull bge-m3
```

## 运行顺序

```bash
# 1. 分块策略对比实验 (会分别建 chroma_db_fixed/ 和 chroma_db_sliding/ 两个库)
python experiments/run_chunking_compare.py

# 2. 混合检索对比实验 (会建 chroma_db_hybrid/)
python experiments/run_hybrid_compare.py

# 3. 聊天记录记忆抽取
python -m src.memory_extract
```

