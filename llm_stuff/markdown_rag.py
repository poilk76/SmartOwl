from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from pathlib import Path
from langchain.tools import tool
import os

class Knowlage_Base:

    def __init__(
            self,
            path:Path=Path('./db/'),
            embedding_model:OllamaEmbeddings = OllamaEmbeddings(model='embeddinggemma:300m')
            ):

        self.db_path = path
        self.embedding_model = embedding_model

        if os.path.exists(self.db_path):
            self.db = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_model
                )
        else:
            self.db = None



    def __chunking(self,documents: list[Document]) -> list[Document]:

        result = []

        headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=100
        )

        for document in documents:

            path = document.metadata['source']

            chunks = markdown_splitter.split_text(document.page_content)

            for chunk in chunks:
                text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)

                if len(text) > 1100:
                    result = result + [Document(page_content=content,\
                                                metadata={'source':path})\
                                                for content in text_splitter.split_text(text)]
                else:
                    chunk.metadata = {'source':path}
                    result.append(chunk)

        return result



    def __embedding(self,chunks: list[Document]) -> None:

        if self.db == None:

            self.db = Chroma.from_documents(
                chunks, 
                self.embedding_model,
                persist_directory=self.db_path
                )
        
        else:

            self.db.from_documents(
                chunks
                )


    def load_documents(self,documents:list[Document]) -> None:

        chunks = self.__chunking(documents)

        self.__embedding(chunks)

    @tool
    def search(self,query:str,amount:int=3) -> list[Document]:
        """Search the user's indexed knowledge base using semantic retrieval. 
        Returns the most relevant passages from user-provided documents to 
        ground responses in retrieved context."""

        db = Chroma(persist_directory=self.db_path, embedding_function=self.embedding_model)

        result = db.similarity_search_with_relevance_scores(query,k=amount)

        return f'data:query/result;result,{result}'


