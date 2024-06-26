from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_community import embeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter

llm_local = ChatOllama(model="qwen:7b")

# 1，读取文件并分词
documents = TextLoader("./3guo2.txt").load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(documents)

# 2，嵌入并存储
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = DocArrayInMemorySearch.from_documents(doc_splits, embeddings)
retriever = vectorstore.as_retriever()

# 3，向模型提问
template = """
Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm_local
    | StrOutputParser())
resp = chain.invoke("身长七尺，细眼长髯的是谁？")
print(resp)

