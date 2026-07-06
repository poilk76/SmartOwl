from langchain.tools import tool
from mimetypes import guess_type
from base64 import b64encode
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=1000
)

@tool
def get_image(path:str) -> str:
    """Load a local image and return it as a base64 data URI."""

    mime = guess_type(path)[0] or "image/png"
    
    with open(path,'rb') as file:

        image = b64encode(file.read()).decode()

    return f"data:{mime};base64,{image}"


@tool
def search_wikipedia(query:str) -> str:
    """
    Search Wikipedia for information about a topic.

    Use this tool when you need general world knowledge, historical facts,
    biographies, scientific concepts, places, organizations, or other
    encyclopedic information.
    """
    return wikipedia.run(query)