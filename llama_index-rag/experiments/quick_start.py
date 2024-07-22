"""
这是一个最简单的示例， 展示了构建一个RAG的全流程: 加载文档， 构建索引， 构建query engine， 最后进行查询。
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

###定义使用的llm，embedding 模型
llm = Ollama(model="qwen2:7b")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

## 加载文档
documents = SimpleDirectoryReader("../../data").load_data()
print("documents: ", len(documents))

## 构建index
index = VectorStoreIndex.from_documents(documents, show_progress=True)

## 构建query engine
query_engine = index.as_query_engine()
query = "身长九尺，髯长二尺的人是谁？"

## query
response = query_engine.query(query)
print(f"query:{query}")
print(f"查询结果：{response}")
