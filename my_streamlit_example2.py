import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import streamlit as st

st.title('🦜🔗 搜索助手')

@tool
def search_web(query: str) -> str:
    """使用 Tavily 搜索互联网获取最新信息"""
    from langchain_community.tools.tavily_search import TavilySearchResults
    search = TavilySearchResults(max_results=3)
    return search.run(query)

tools = [search_web]

if prompt := st.chat_input("请输入您的问题:"):
    llm = ChatOpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ.get("Deepseek_API_KEY"),
        model="deepseek-chat",
        temperature=0.7
    )

    agent = create_react_agent(llm, tools)

    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        st.write(response["messages"][-1].content)