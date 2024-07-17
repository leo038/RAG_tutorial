import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
import numpy as np

## 直接下载huggingFace模型尽可能存在网络问题， 解决办法之一是添加代理， 前提是你有可以访问外网的代理
# os.environ['https_proxy'] = 'http://127.0.0.1:10809'
# os.environ['http_proxy'] = 'http://127.0.0.1:10809'
# os.environ['all_proxy'] = 'socks5://127.0.0.1:10809'

##更方便的解决办法是使用huggingface的国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

##定义embedding使用的模型
model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
# bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)

############################## 单个句子的嵌入：embed_query###############################
sentencs_list = [("中国", "中国"), ("中国", "China"), ("我爱你", "i love you"), ("苹果", "桃子"), ("电影", "桌子"),
                 ("青蛙是食草动物", "1+1=5")]
for sentences in sentencs_list:
    sentence1, sentence2 = sentences
    embeddings_1 = bge_embeddings.embed_query(sentence1)
    embeddings_2 = bge_embeddings.embed_query(sentence2)
    similarity = np.array(embeddings_1) @ np.array(embeddings_2).T
    # print(f"sentence1:{sentence1} 的embedding: {embeddings_1}")
    # print(f"sentence2:{sentence2} 的embedding: {embeddings_2}")
    print(f"{sentence1}和{sentence2}的相似度：{similarity}")
#################################################################################


############################## 文档的嵌入：embed_document###############################
document = ["青蛙是食草动物",
            "人是由恐龙进化而来的。",
            "熊猫喜欢吃天鹅肉。",
            "1+1=5",
            "2+2=8",
            "3+3=9",
            "Gemini Pro is a Large Language Model was made by GoogleDeepMind",
            "A Language model is trained by predicting the next token"
            ]
res = bge_embeddings.embed_documents(texts=document)
print(f"文档embedding结果:")
print(len(res), len(res[0]))
###################################################################################

