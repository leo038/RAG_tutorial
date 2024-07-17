"""
document transform的主要作用是对文档进行分割， 组合， 过滤等等操作
可参考 lagnchain代码中的 libs/community/langchain_community/document_transformers
一般对文档进行分割用的最多， 可参考 libs/langchain/langchain/text_splitter.py
"""

with open("data/test.txt") as f:
    read_file = f.read()

#####################1 简单的基于字符的分割###################################
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    # separator="\n\n",  # 指定了用于分割文本的分隔符为两个连续的换行符
    chunk_size=20,  # 指定了每个分割后的文本块的大小为20个字符
    chunk_overlap=10,  # 指定了相邻文本块之间的重叠大小为10个字符，避免信息丢失
    # length_function=len,  # 指定了计算字符串长度的函数为内置的len函数
    # is_separator_regex=False,  # 指定了分隔符不是一个正则表达式

)

texts = text_splitter.create_documents([read_file])
print("简单的基于字符的分割结果：")
for index, text in enumerate(texts):
    print(index, text)
##################################################################


#####################2 字符递归分割割###################################
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,  # 块大小（每个分割文本的字符数量）
    chunk_overlap=10,  # 块重叠（两个相邻块之间重叠的字符数量）
    length_function=len,  # 长度函数（用于计算文本长度的函数）
    add_start_index=True,  # 添加起始索引（是否在结果中包含分割文本的起始索引）
)

texts = text_splitter.create_documents([read_file])
print("字符递归分割的结果：")
for index, text in enumerate(texts):
    print(index, text)
##################################################################
