import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. 页面基础配置 (必须是第一个 Streamlit 命令)
st.set_page_config(page_title="智能多工具助手", page_icon="🦜🔗", layout="wide")
load_dotenv()

# 2. 侧边栏配置面板
with st.sidebar:
    st.header("⚙️ 模型与设置")
    # 允许用户在界面上动态输入或修改 API Key
    api_key = st.text_input("DeepSeek API Key", value=os.environ.get("Deepseek_API_KEY", ""), type="password")
    model_name = st.selectbox("选择模型版本", ["deepseek-chat", "deepseek-coder"])
    temperature = st.slider("Temperature (发散度)", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    
    st.markdown("---")
    st.markdown("### 🔧 已挂载工具")
    st.checkbox("Tavily 互联网搜索", value=True, disabled=True)
    st.checkbox("Python 表达式计算", value=True, disabled=True)
    
    st.markdown("---")
    # 清空上下文历史功能
    if st.button("🗑️ 清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title('🦜🔗 智能多工具对话助手')
st.caption("集成了 Web 搜索与数学计算能力的 LangGraph Agent")

# 3. 定义多个工具 (Tools)
@tool
def search_web(query: str) -> str:
    """使用 Tavily 搜索互联网获取最新信息"""
    search = TavilySearchResults(max_results=3)
    return search.run(query)

@tool
def calculate(expression: str) -> str:
    """安全的数学计算器，用于计算复杂的数学表达式（如：23 * 45 / 2.1）"""
    try:
        # 限制计算环境以保证代码安全
        allowed_names = {"__builtins__": None}
        return str(eval(expression, allowed_names, {}))
    except Exception as e:
        return f"计算错误: {str(e)}"

# 将工具组装成列表
tools = [search_web, calculate]

# 4. 初始化 Session State，用于维持多轮对话上下文
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. 渲染历史对话记录
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# 6. 处理用户输入与交互
if prompt := st.chat_input("请输入您的问题或指令..."):
    # 校验 API Key
    if not api_key:
        st.warning("请在左侧边栏输入您的 DeepSeek API Key！")
        st.stop()
        
    # a. 将用户问题追加到历史记录并渲染在界面上
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)
    
    # b. 初始化 LLM 模型 (修复了原代码缺少的括号)
    llm = ChatOpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=api_key,
        model=model_name,
        temperature=temperature
    )
    
    # c. 创建 LangGraph React Agent
    agent = create_react_agent(llm, tools)
    
    # d. 执行 Agent 并捕获输出
    with st.chat_message("assistant"):
        # 使用 st.status 提供“思考中”的动画交互效果
        with st.status("Agent 正在思考和调用工具...", expanded=True) as status:
            try:
                # 将完整的消息历史传入，赋予大模型上下文记忆能力
                response = agent.invoke({"messages": st.session_state.messages})
                final_msg = response["messages"][-1]
                
                status.update(label="执行完成！", state="complete", expanded=False)
                
                # 渲染结果
                st.write(final_msg.content)
                
                # 将助手的最终回答追加到历史记录中
                st.session_state.messages.append(final_msg)
                
            except Exception as e:
                status.update(label="执行过程中发生错误", state="error")
                st.error(f"调用失败: {str(e)}")
