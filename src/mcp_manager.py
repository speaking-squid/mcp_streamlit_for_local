from typing import Optional, Dict, Any, List
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import os

SYSTEM_PROMPT = """<ROLE>
You are a smart agent with an ability to use tools. 
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question. 
If you are failed to answer the question, try different tools to get context.
Your answer should be very polite and professional.
</ROLE>

----

<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question is consist of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- If you are failed to answer the question, try different tools to get context.

Step 3: Answer the question
- Answer the question in the same language as the question.
- Your answer should be very polite and professional.

Step 4: Provide the source of the answer(if applicable)
- If you've used the tool, provide the source of the answer.
- Valid sources are either a website(URL) or a document(PDF, etc).

Guidelines:
- If you've used the tool, your answer should be based on the tool's output(tool's output is more important than your own knowledge).
- If you've used the tool, and the source is valid URL, provide the source(URL) of the answer.
- Skip providing the source if the source is not URL.
- Answer in the same language as the question.
- Answer should be concise and to the point.
- Avoid response your output with any other information than the answer and the source.  
</INSTRUCTIONS>

----

<OUTPUT_FORMAT>
(concise answer to the question)

**Source**(if applicable)
- (source1: valid URL)
- (source2: valid URL)
- ...
</OUTPUT_FORMAT>
"""

class MCPManager:
    def __init__(self, config_path: str = "./src/config.json"):
        self.config_path = config_path
        self.mcp_client = None
        self.agent = None
        self.memory = MemorySaver()
        self.tool_list_each_server = None
        try:
            self._config = {} if not os.path.exists(config_path) else json.load(open(config_path, "r", encoding="utf-8"))
        except json.JSONDecodeError:
            raise Exception(f"설정 파일이 올바른 JSON 형식이 아닙니다.")
        except Exception as e:
            raise Exception(f"설정 파일 로드 중 오류 발생: {str(e)}")
    
    def load_config(self) -> Dict[str, Any]:
        """MCP 설정을 로드합니다."""
        return self._config
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """MCP 설정을 저장합니다."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            self._config = config
            return True
        except Exception as e:
            raise Exception(f"설정 파일 저장 중 오류 발생: {str(e)}")
    
    def add_tool(self, tool_config: Dict[str, Any]) -> bool:
        """새로운 MCP 도구를 추가합니다."""
        config = self.load_config()

        if "mcpServers" in tool_config:
            for name, server_config in tool_config["mcpServers"].items():
                config[name] = server_config
        else:
            for name, server_config in tool_config.items():
                config[name] = server_config
        return self.save_config(config)
    
    def remove_tool(self, tool_name: str) -> bool:
        """MCP 도구를 제거합니다."""
        config = self.load_config()
        if tool_name in config:
            del config[tool_name]
            return self.save_config(config)
        return False
    
    def get_tools(self) -> List[str]:
        """등록된 MCP 도구 목록을 반환합니다."""
        config = self.load_config()
        return list(config.keys())
    
    def get_tool_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """특정 MCP 도구의 설정을 반환합니다."""
        config = self.load_config()
        return config.get(tool_name)
    
    async def initialize(self, llm):
        """MCP 클라이언트와 에이전트를 초기화합니다."""
        config = self.load_config()
        self.mcp_client = MultiServerMCPClient(self.load_config())
        try:
            await self.mcp_client.__aenter__()
        except Exception as e:
            print(f"error: {e}")
            import traceback
            traceback.print_exc()
            raise
        tools = self.mcp_client.get_tools()
        self.tool_list_each_server = self.mcp_client.server_name_to_tools
        self.agent = create_react_agent(llm, tools, checkpointer=MemorySaver() ,prompt=SYSTEM_PROMPT)

        return True
    
    def get_tools_from_mcp_server(self, server_name: str) -> List[str]:
        """MCP 클라이언트에서 도구 목록을 반환합니다."""
        tools = self.tool_list_each_server.get(server_name, [])
        return [tool.name for tool in tools]
        
    def get_agent(self):
        """MCP 에이전트를 반환합니다."""
        return self.agent

    def get_client(self):
        """MCP 클라이언트를 반환합니다."""
        return self.mcp_client
    
    async def server_close(self):
        """MCP 서버를 종료합니다."""
        if self.mcp_client is not None:
            await self.mcp_client.__aexit__(None, None, None)
            self.mcp_client = None       
        return True
