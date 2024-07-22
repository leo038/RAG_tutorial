"""
文档加载器的作用是加载文档， 包括本地的文档和网络上的资源等等。
langchain中集成了非常多的文档加载器， 涵盖常见的文档格式， 以及网络资源， 如网页， 博客， youtube等等
具体可参考langchain代码库 libs/community/langchain_community/document_loaders/__init__.py
"""


## CSV 加载
from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path="../../data/data1.csv", encoding="utf-8")
data = loader.load()
print(data)


## txt 加载
from langchain.document_loaders import TextLoader

loader = TextLoader(file_path='.../../data/test.txt', encoding="utf-8")
data = loader.load()
print(data)


## markdown文件加载
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# import nltk
# nltk.download('punkt')
loader = UnstructuredMarkdownLoader(file_path="../../data/README.md")
data = loader.load()
print(data)