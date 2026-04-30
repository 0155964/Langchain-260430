import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

st.title('🦜🔗 中文小故事生成器')

def get_api_key():
    return os.environ.get("Deepseek_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

prompt = ChatPromptTemplate.from_template("请编写一篇关于{topic}的中文小故事，不超过100字")

model = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    api_key=get_api_key(),
    model="deepseek-chat",
    temperature=0.8
)

chain = prompt | model

with st.form('my_form'):
    text = st.text_area('输入主题关键词:', '小白兔')
    submitted = st.form_submit_button('提交')
    if submitted:
        st.info(chain.invoke({"topic": text}))