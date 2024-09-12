from pydantic import BaseModel
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings


class NewsArticle(BaseModel):

    id: int
    headline: str
    summary: str
    # source: str
    # created_at: str

    def to_document(self):

        doc_id = hashlib.md5(self.headline.encode()).hexdigest()
        document = Document(id=doc_id)
        document.text = [self.summary, self.headline]

        return document


class Document(BaseModel):

    id: str
    text: list = []
    chunks: list = []
    embeddings: list = []

    def compute_chunks(self):
        for item in self.text:
            recursive_text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=20,
                length_function=len,
                is_separator_regex=False,
            )
            recursive_text_splitter_chunks = recursive_text_splitter.split_text(
                item
            )
            self.chunks.extend(recursive_text_splitter_chunks)
        return self

    def compute_embeddings(self):
        ollama_emb = OllamaEmbeddings(model="all-minilm:l6-v2")
        embedding = ollama_emb.embed_documents(self.chunks)
        self.embeddings.append(embedding)

        return self

    def to_payloads(self):
        payloads = []
        ids = []

        for chunk in self.chunks:
            payload = {"text": chunk}
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()

            payloads.append(payload)
            ids.append(chunk_id)

        return chunk_id, payload
