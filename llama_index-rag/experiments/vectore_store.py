"""
这个示例展示了把构建的索引存入向量数据库中进行保存， 以及如何从已经保存的向量数据库中加载索引
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
import os

###定义使用的llm，embedding 模型
llm = Ollama(model="qwen2:7b")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

## 构建index
# initialize client, setting path to save data
vectore_store_dir = "./chroma_db"
db = chromadb.PersistentClient(path=vectore_store_dir)

# create collection
chroma_collection = db.get_or_create_collection("quickstart")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

## 如果已经存入过数据库，则直接从数据库文件中加载index, 否则从文档中构建index, 并存入数据库
##db = chromadb.PersistentClient(path=vectore_store_dir) 时已经生成了目录并且有一个默认文件chroma.sqlite3， 所以不能通过有相应的文件路径判断， 而应该判断是否已经生成了实际的数据库文件
if len(os.listdir(vectore_store_dir)) > 1:
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
else:
    ## 加载文档
    documents = SimpleDirectoryReader("../../data").load_data()
    print("documents: ", len(documents))

    index = VectorStoreIndex.from_documents(documents, show_progress=True, storage_context=storage_context)

## 构建query engine
query_engine = index.as_query_engine()
query = "身长九尺，髯长二尺的人是谁？"

## query
response = query_engine.query(query)
print(f"query:{query}")
print(f"查询结果：{response}")
