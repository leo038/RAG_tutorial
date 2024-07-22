"""
检索的作用主要是根据输入的查询信息， 找到相关的文档片段。 这一步主要是完成了传统的文档检索功能。
RAG中有2个步骤： 第一步完成传统的检索， 找到相关的文档片段； 第二步会把找到的文档片和原始输入的查询信息组合在一起， 送给大模型， 得到最终的输出结果。
langchain中检索的概念实际上包含了这2个步骤的内容， 要注意区分。
langchain中提供的检索器具体可参考： libs/langchain/langchain/retrievers
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

## 普通的检索器
bge_retriever = vectore_data.as_retriever(search_kwargs={"k": 1})  # k控制检索多少条最相关的，
res = bge_retriever.get_relevant_documents(query=query)
print(f"query:{query}, 检索到的内容：{res}")

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import Ollama

llm = Ollama(model="llama3:8b")
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=bge_retriever, llm=llm
)

unique_docs = retriever_from_llm.get_relevant_documents(query=query)
print(f"带llm的检索器检索到的内容：{unique_docs}")
