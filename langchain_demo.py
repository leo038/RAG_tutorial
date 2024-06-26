from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm_local = ChatOllama(model="qwen:7b")
template = "{topic}"
prompt = ChatPromptTemplate.from_template(template)
chain = llm_local | StrOutputParser()
resp = chain.invoke("身长七尺，细眼长髯的是谁？")
print(resp)

