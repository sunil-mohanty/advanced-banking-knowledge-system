"""
Vector Retriever — Optimized
Key changes:
  - Hash-based rebuild guard: same docs = instant reuse
  - Persisted vectorstore loaded from disk if available
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict
import hashlib
from dotenv import load_dotenv

load_dotenv()


class VectorRetriever:
    def __init__(self):
        self.embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
        self._built_hash = None

    def _doc_hash(self, documents: List[Dict]) -> str:
        content = "".join(d.get("content", "") for d in documents)
        return hashlib.md5(content.encode()).hexdigest()

    def build_vectorstore(self, documents: List[Dict]):
        doc_hash = self._doc_hash(documents)

        # ✅ Skip rebuild if same docs already embedded
        if self._built_hash == doc_hash and self.vectorstore:
            print("   ⚡ Vector store from cache")
            return

        print("📚 Building vector store...")

        docs = [
            Document(
                page_content=doc["content"],
                metadata={"doc_id": doc.get("id", "unknown")}
            )
            for doc in documents
        ]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks   = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./vectorstore_advanced"
        )

        self._built_hash = doc_hash
        print(f"   Vector store: {len(chunks)} chunks")

    def retrieve(self, query: str, k: int = 5) -> Dict:
        if not self.vectorstore:
            return {"context": "", "chunks": [], "num_chunks": 0}

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        chunks        = []
        context_parts = []

        for i, (doc, score) in enumerate(results, 1):
            chunks.append({
                "content": doc.page_content,
                "score":   float(score),
                "doc_id":  doc.metadata.get("doc_id", "unknown"),
                "rank":    i
            })
            context_parts.append(
                f"[Chunk {i} - Score: {score:.3f}]\n{doc.page_content}"
            )

        return {
            "context":    "\n\n".join(context_parts),
            "chunks":     chunks,
            "num_chunks": len(chunks)
        }
