import streamlit as st
import asyncio
import nest_asyncio
import os
import platform
import json
from mcp_manager import MCPManager
from llm import LLMProvider
from langchain_core.messages import AIMessageChunk, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from utils import astream_graph, random_uuid

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
nest_asyncio.apply()

if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

st.set_page_config(
    page_title="MCP Chat Agent",
    page_icon="🤖",
    layout="wide",
)



# ----------- 대화 진행 관련 함수 : https://github.com/teddynote-lab/langgraph-mcp-agents 참조

def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    Creates a streaming callback function.

    This function creates a callback function to display responses generated from the LLM in real-time.
    It displays text responses and tool call information in separate areas.

    Args:
        text_placeholder: Streamlit component to display text responses
        tool_placeholder: Streamlit component to display tool call information

    Returns:
        callback_func: Streaming callback function
        accumulated_text: List to store accumulated text responses
        accumulated_tool: List to store accumulated tool call information
    """
    accumulated_text = []
    accumulated_tool = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        # print(f"message: {message}\n")
        message_content = message.get("content", None)

        if isinstance(message_content, AIMessageChunk):
            content = message_content.content
            # If content is in list form (mainly occurs in Claude models)
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                # Process text type
                if message_chunk["type"] == "text":
                    accumulated_text.append(message_chunk["text"])
                    text_placeholder.markdown("".join(accumulated_text))
                # Process tool use type
                elif message_chunk["type"] == "tool_use":
                    if "partial_json" in message_chunk:
                        accumulated_tool.append(message_chunk["partial_json"])
                    else:
                        tool_call_chunks = message_content.tool_call_chunks
                        tool_call_chunk = tool_call_chunks[0]
                        accumulated_tool.append(
                            "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                        )
                    with tool_placeholder.expander(
                        "🔧 Tool Call Information", expanded=True
                    ):
                        st.markdown("".join(accumulated_tool))
            # Process if tool_calls attribute exists (mainly occurs in OpenAI models)
            elif (
                hasattr(message_content, "tool_calls")
                and message_content.tool_calls
                and len(message_content.tool_calls[0]["name"]) > 0
            ):
                tool_call_info = message_content.tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # Process if content is a simple string
            elif isinstance(content, str):
                accumulated_text.append(content)
                text_placeholder.markdown("".join(accumulated_text))
            # Process if invalid tool call information exists
            elif (
                hasattr(message_content, "invalid_tool_calls")
                and message_content.invalid_tool_calls
            ):
                tool_call_info = message_content.invalid_tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information (Invalid)", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # Process if tool_call_chunks attribute exists
            elif (
                hasattr(message_content, "tool_call_chunks")
                and message_content.tool_call_chunks
            ):
                tool_call_chunk = message_content.tool_call_chunks[0]
                accumulated_tool.append(
                    "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                )
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # Process if tool_calls exists in additional_kwargs (supports various model compatibility)
            elif (
                hasattr(message_content, "additional_kwargs")
                and "tool_calls" in message_content.additional_kwargs
            ):
                tool_call_info = message_content.additional_kwargs["tool_calls"][0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 Tool Call Information", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
        # Process if it's a tool message (tool response)
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append(
                "\n```json\n" + str(message_content.content) + "\n```\n"
            )
            with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                st.markdown("".join(accumulated_tool))
        return None
    
    return callback_func, accumulated_text, accumulated_tool

async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """
    Processes user questions and generates responses.

    This function passes the user's question to the agent and streams the response in real-time.
    Returns a timeout error if the response is not completed within the specified time.

    Args:
        query: Text of the question entered by the user
        text_placeholder: Streamlit component to display text responses
        tool_placeholder: Streamlit component to display tool call information
        timeout_seconds: Response generation time limit (seconds)

    Returns:
        response: Agent's response object
        final_text: Final text response
        final_tool: Final tool call information
    """
    agent = st.session_state.mcp_manager.get_agent()
    if not agent:
        raise Exception("MCP 에이전트가 초기화되지 않았습니다.")
    
    # print(f"agent: {agent.get_graph()}")

    try:
        streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
            get_streaming_callback(text_placeholder, tool_placeholder)
        )
        try:
            response = await asyncio.wait_for(
                astream_graph(
                    agent,
                    {"messages": [HumanMessage(content=query)]},
                    callback=streaming_callback,
                    config=RunnableConfig(
                        recursion_limit=100,
                        thread_id=st.session_state.thread_id
                    ),
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            error_msg = f"⏱️ Request time exceeded {timeout_seconds} seconds. Please try again later."
            return {"error": error_msg}, error_msg, ""

        final_text = "".join(accumulated_text_obj)
        final_tool = "".join(accumulated_tool_obj)
        return response, final_text, final_tool
    except Exception as e:
        import traceback

        error_msg = f"❌ Error occurred during query processing: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""

# ----------- 대화 진행 관련 함수 : https://github.com/teddynote-lab/langgraph-mcp-agents 참조

async def cleanup_mcp_servers():
    """MCP 서버를 종료한다"""
    if st.session_state.mcp_manager is not None:
        await st.session_state.mcp_manager.server_close()
        st.session_state.mcp_manager = None

async def init_agent():
    """mcp agent를 초기화한다"""
    with st.spinner("MCP 서버에 연결 중..."):
        await cleanup_mcp_servers()
        llm = st.session_state.llm_provider.get_llm()
        st.session_state.mcp_manager = MCPManager()
        success = await st.session_state.mcp_manager.initialize(llm)
        if success:
            st.success("MCP 에이전트가 초기화되었습니다.")
        else:
            st.error("MCP 에이전트 초기화 실패")
        return True

def apply_agent():
    """mcp agent를 적용한다"""
    try:
        success = st.session_state.event_loop.run_until_complete(init_agent())
        if success:
            st.success("설정이 적용되었습니다.")
        else:
            st.error("설정 적용 실패")
    except Exception as e:
        st.error(f"설정 적용 중 오류 발생: {e}")


if "session_initialized" not in st.session_state or not st.session_state.session_initialized:
    st.session_state.session_initialized = True
    st.session_state.messages = []
    st.session_state.llm_provider = None
    st.session_state.mcp_manager = None
    st.session_state.thread_id = random_uuid()
    st.session_state.timeout = 60

# 사이드바 설정 - LLM 설정, MCP 도구 설정
with st.sidebar:
    st.title("MCP Chat Agent")
    st.write("MCP 도구 사용 채팅 에이전트")

    llm_provider = st.selectbox("LLM Provider",["OpenAI", "Azure_OpenAI", "Anthropic"])
    api_key = st.text_input("API Key", type="password")
    
    # 특정 모델에 대한 특정 엔드포인트 설정
    endpoint = ""
    if llm_provider in  ["Azure_OpenAI", "OpenAI"]:
        if llm_provider == "OpenAI":
            endpoint = "https://api.openai.com/v1"
        endpoint = st.text_input(f"{llm_provider} Endpoint", value=endpoint)

    # 모델 기본값 설정
    model_value = ""
    if llm_provider in ["Azure_OpenAI", "OpenAI"]:
        model_value = "gpt-4o"
    elif llm_provider == "Anthropic":
        model_value = "claude-3-opus-20240229"
    model = st.text_input(f"{llm_provider} model", value=model_value)

    if api_key and (True if llm_provider == "Anthropic" else endpoint) and model:
        os.environ[f"{llm_provider.upper()}_API_KEY"] = api_key
        if st.session_state.llm_provider is None or \
            st.session_state.llm_provider.get_provider() != llm_provider or \
            st.session_state.llm_provider.get_model() != model or \
            st.session_state.llm_provider.get_endpoint() != endpoint:            
            st.session_state.llm_provider = LLMProvider(llm_provider, endpoint, model)
            if endpoint == "http://localhost:11434/v1":
                st.session_state.timeout = 1200
            apply_agent()

    st.divider()
    st.subheader("MCP 도구 관리")

    with st.expander("MCP 도구 추가"):
        tool_config = st.text_area("도구 설정(JSON)",
                                   value="",
                                   height=200)
        if st.button("도구 추가"):
            try:
                config = json.loads(tool_config)
                if st.session_state.mcp_manager.add_tool(config):
                    st.success(f"도구 추가 완료")
                    apply_agent()
                else:
                    st.error("도구 추가 실패")
            except json.JSONDecodeError:
                st.error("유효하지 않은 JSON 형식입니다.")
            except Exception as e:
                st.error(f"도구 추가 중 오류 발생: {e}")
            
    st.subheader("등록 도구 목록")
    # with st.expander("등록 도구 목록"):
    if st.session_state.mcp_manager:
        tools = st.session_state.mcp_manager.get_tools()
        if tools:
            for tool in tools:
                tool_count = len(st.session_state.mcp_manager.get_tools_from_mcp_server(tool))
                with st.expander(f"{tool} - {tool_count} 건"):
                    subtools = st.session_state.mcp_manager.get_tools_from_mcp_server(tool)
                    if subtools:
                        st.markdown("**도구 목록:**")
                        for subtool in subtools:
                            st.write(f"- {subtool}")
                    else:
                        st.write("등록된 도구가 없습니다.")
                    if st.button("삭제", key=f"delete_{tool}"):
                        if st.session_state.mcp_manager.remove_tool(tool):
                            st.success(f"도구 '{tool}' 삭제 완료")
                            apply_agent()
                            st.rerun()
                        else:
                            st.error(f"도구 '{tool}' 삭제 실패")
        else:
            st.info("등록된 도구가 없습니다.")
                    
    # 너무 잦은 apply_agent 호출이 되는 경우 설적 적용 버튼 클릭 시에만 적용되도록 할 수 있음
    #if st.button("설정 적용", type="primary"):
    #    apply_agent()


st.title("MCP Chat Agent")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("메시지를 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        tool_placeholder = st.empty()
        text_placeholder = st.empty()
        with st.spinner("생각중..."):
            if st.session_state.mcp_manager is None:
                response = "LLM Provider와 API Key를 설정해주세요."
            else:
                try:
                    response, final_text, final_tool = (
                        st.session_state.event_loop.run_until_complete(
                            process_query(prompt, text_placeholder, tool_placeholder, st.session_state.timeout)
                        )
                    )
                except Exception as e:
                    response = f"오류 발생: {e}"
                st.session_state.messages.append(
                    {"role" : "assistant", "content" : final_text}
                )

                if final_tool.strip():
                    st.session_state.messages.append(
                        {
                            "role" : "assistant_tool",
                            "content" : final_tool
                        }
                    )
                st.rerun()



                
