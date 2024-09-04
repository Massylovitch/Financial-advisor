from pydantic import BaseModel
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

class NewsArticle(BaseModel):

    id: int
    headline:str
    summary: str
    # source: str
    # created_at: str

    def to_document(self):

        doc_id = hashlib.md5(self.headline.encode()).hexdigest()
        document = Document(id=doc_id)
        document.text = [self.summary]

        return document

class Document(BaseModel):

    id: str
    text: list = []
    chunks: list = []
    embeddings: list = []

    def compute_chunks(self):
        recursive_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )

        recursive_text_splitter_chunks = recursive_text_splitter.create_documents(self.text)
        self.chunks.extend(recursive_text_splitter_chunks)


    def compute_embeddings(self):
        ollama_emb  = OllamaEmbeddings(model="llama3")
        embedding  = ollama_emb.embed_documents(self.chunks)
        self.embeddings.append(embedding)
        print(self.embeddings)