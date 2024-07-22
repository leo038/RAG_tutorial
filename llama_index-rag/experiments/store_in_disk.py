"""
这个示例展示了把构建的索引存入磁盘中进行持久化保存， 以及如何从已经存储的文件中加载索引
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import os

###定义使用的llm，embedding 模型
llm = Ollama(model="qwen2:7b")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

#########################构建index####################################
persist_dir = "./cache"
if os.path.exists(persist_dir):
    ##从磁盘中已保存的数据中加载index
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
else:
    ## 加载文档
    documents = SimpleDirectoryReader("../../data").load_data()
    print("documents: ", len(documents))

    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    ## 在这里把构建好的index存入磁盘中持久化保存
    index.storage_context.persist(persist_dir=persist_dir)

#####################################################################

## 构建query engine
query_engine = index.as_query_engine()
query = "身长九尺，髯长二尺的人是谁？"

## query
response = query_engine.query(query)
print(f"query:{query}")
print(f"查询结果：{response}")
