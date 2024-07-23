"""
这个示例展示了如何进行自定义的文档切割策略， 不同的切割策略如何影响最终的输出效果。
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter

###定义使用的llm，embedding 模型
llm = Ollama(model="gemma2:9b", request_timeout=500)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

## 加载文档
documents = SimpleDirectoryReader("../../data").load_data()
print("documents: ", len(documents))

## 构建index
##############################################################################
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=10)   #chunk_size取512时， 包含正确答案的文本片段刚好被切成2部分， 模型的回答经常不正确
Settings.text_splitter = text_splitter
index = VectorStoreIndex.from_documents(documents, show_progress=True, transformations=[text_splitter])
###########################################################################


## 构建query engine
query_engine = index.as_query_engine()
query = "身长九尺，髯长二尺的人是谁？"
# query = "what is crewAI？"

## query
# 采用llm = Ollama(model="qwen2:7b")时， 结果一直是GG。 进一步发现， 直接使用ollama 运行qwen2:7b时， 不管输入什么， 输出都是GG, 应该是模型本身问题
# 采用llm = Ollama(model="qwen:7b") 回答正确
# 采用llm = Ollama(model="mistral:7b") 回答正确
# 采用llm = Ollama(model="llama3:8b")回答正确
# 采用llm = Ollama(model="llama2:13b")回答正确
# 采用llm = Ollama(model="gemma2:9b")时， 无法访问Ollama对应的服务。 直接用ollama run 也报错， 应该是模型本身的问题
# 注意： 回答的结果比较依赖于构建索引是的文档切割策略， 切割的不好回答经常不对
response = query_engine.query(query)
print(f"query:{query}")
print(f"查询结果：{response}")




### 上面直接通过query_engine出来的结果经常不对， 通过下面的方式验证一下检索的内容是否正确
# 发现检索出来的结果是正确的， 那么问题应该是出在最后大模型整合输出上了
retriever = index.as_retriever()
res = retriever.retrieve(query)
print(f"中间检索结果：{res}")
