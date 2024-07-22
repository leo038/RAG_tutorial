"""
向量存储的目标是使用向量数据库存储文档的embedding信息， 便于后续进行高效的检索
langchain中提供了丰富的向量数据库接口，可参考： libs/langchain/langchain/vectorstores
"""
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings

# 加载txt文档
loader = TextLoader(file_path='../../data/test.txt', encoding="utf-8")
raw_documents = loader.load()

# 分割文本
text_splitter = CharacterTextSplitter(chunk_size=7500, chunk_overlap=100)

documents = text_splitter.split_documents(raw_documents)
print(f"分割后的文档： {documents}")

# 将分割后的文本，使用 BAAI的bge模型得到embedding, 并用Chroma向量数据库进行存储
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
vectore_data = Chroma.from_documents(documents=documents, embedding=bge_embeddings)

query = "身长九尺，髯长二尺的人是谁"
docs = vectore_data.similarity_search(query, k=1)  # 注意： 文本分割的时候chunk_size=7500, chunk_overlap=100如果设置不合适， 可能检索不到正确的文本片段
print(f"query:{query}, 从向量数据库中找到的最相关的文档片段：")
print(len(docs), docs)
