from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import os

class LLMProvider:
    def __init__(self, provider: str = "OpenAI", endpoint: str = None, model: str = None):
        self.provider = provider
        self._llm = None
        self.endpoint = endpoint
        self.model = model
    def get_llm(self):
        """선택된 provider에 따라 LLM 인스턴스를 반환합니다."""
        if self._llm is None:
            if self.provider == "OpenAI":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key가 설정되지 않았습니다.")
                self._llm = ChatOpenAI(
                    model=self.model,
                    temperature=0.7,
                    streaming=True,
                    base_url=self.endpoint
                )
            elif self.provider == "Anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("Anthropic API key가 설정되지 않았습니다.")
                self._llm = ChatAnthropic(
                    model=self.model,
                    temperature=0.7,
                    streaming=True
                )
            elif self.provider == "Azure_OpenAI":
                api_key = os.getenv("AZURE_OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("Azure API key가 설정되지 않았습니다.")
                self._llm = AzureChatOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=api_key,
                    api_version="2025-01-01-preview",
                    deployment_name=self.model,
                    temperature=0.7,
                    streaming=True
                )
            else:
                raise ValueError(f"지원하지 않는 LLM provider입니다: {self.provider}")
        return self._llm
    
    def get_provider(self):
        return self.provider
    
    def get_model(self):
        return self.model
    
    def get_endpoint(self):
        return self.endpoint
    
    async def generate_response(self, messages: list) -> str:
        """메시지 목록을 받아 LLM 응답을 생성합니다."""
        llm = self.get_llm()
        response = await llm.ainvoke(messages)
        return response.content 