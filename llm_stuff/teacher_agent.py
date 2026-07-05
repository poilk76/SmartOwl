from langchain_ollama import OllamaLLM
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent

llm = OllamaLLM(model='qwen3.5:0.8b')

class Agent_Response(BaseModel):

    summary:str
    sources: list[str]
    tools_used: list[str]

prompt = Chat

agent = create_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)