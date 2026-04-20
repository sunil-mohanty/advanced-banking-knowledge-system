"""
Vector-based Retriever using embeddings and similarity search
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class VectorRetriever:
    """Retrieve using vector similarity search"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
    
    def build_vectorstore(self, documents: List[Dict]):
        """Build vector store from documents"""
        
        print("📚 Building vector store...")
        
        # Convert to LangChain Documents
        docs = []
        for doc in documents:
            docs.append(Document(
                page_content=doc["content"],
                metadata={"doc_id": doc.get("id", "unknown")}
            ))
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(docs)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./vectorstore_advanced"
        )
        
        print(f"   Vector store: {len(chunks)} chunks")
    
    def retrieve(self, query: str, k: int = 5) -> Dict:
        """Retrieve using similarity search"""
        
        print(f"📚 Vector retrieval (top-{k})")
        
        if not self.vectorstore:
            return {"context": "", "chunks": [], "num_chunks": 0}
        
        # Similarity search with scores
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results
        chunks = []
        context_parts = []
        
        for i, (doc, score) in enumerate(results, 1):
            chunk = {
                "content": doc.page_content,
                "score": float(score),
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "rank": i
            }
            chunks.append(chunk)
            
            context_parts.append(
                f"[Chunk {i} - Relevance: {score:.3f}]\n{doc.page_content}"
            )
        
        context = "\n\n".join(context_parts)
        
        print(f"   Found {len(chunks)} relevant chunks")
        
        return {
            "context": context,
            "chunks": chunks,
            "num_chunks": len(chunks)
        }


# Test vector retriever
if __name__ == "__main__":
    
    # Sample documents
    docs = [
        {
            "id": "doc_1",
            "content": "KYC compliance requires verifying customer identity using government ID, proof of address, and SSN. Completion within 30 days is mandatory."
        },
        {
            "id": "doc_2",
            "content": "AML transaction monitoring detects suspicious patterns. Banks file SARs within 30 days and CTRs for cash over $10,000."
        },
        {
            "id": "doc_3",
            "content": "Mortgage loans require 620+ credit score, <43% debt-to-income ratio, and 3-20% down payment."
        }
    ]
    
    # Build and test
    retriever = VectorRetriever()
    retriever.build_vectorstore(docs)
    
    # Test query
    result = retriever.retrieve("What are KYC requirements?", k=3)
    
    print("\n" + "="*70)
    print("VECTOR RETRIEVAL RESULT")
    print("="*70)
    print(result["context"])