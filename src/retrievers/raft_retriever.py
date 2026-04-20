"""
RAFT-style Retriever with document relevance scoring
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict
from dotenv import load_dotenv
import json

load_dotenv()

class RAFTRetriever:
    """RAFT-style retrieval with explicit relevance evaluation"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
    
    def build_vectorstore(self, documents: List[Dict]):
        """Build vector store"""
        
        print("🎯 Building RAFT vector store...")
        
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        docs = [Document(
            page_content=doc["content"],
            metadata={"doc_id": doc.get("id", "unknown")}
        ) for doc in documents]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(docs)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./raft_vectorstore_advanced"
        )
        
        print(f"   RAFT store: {len(chunks)} chunks")
    
    def evaluate_relevance(self, query: str, document: str) -> Dict:
        """Evaluate document relevance using LLM"""
        
        prompt = f"""Evaluate if this document is relevant to answering the query.

Query: {query}

Document: {document}

Evaluate:
1. Is this document relevant? (yes/no/partial)
2. What specific information does it provide?
3. Relevance score (0.0 to 1.0)

Return JSON:
{{
    "is_relevant": "yes/no/partial",
    "relevant_info": "specific information that helps answer query",
    "relevance_score": 0.0-1.0,
    "reasoning": "why this is relevant or not"
}}

JSON:"""
        
        response = self.llm.invoke(prompt)
        
        try:
            evaluation = json.loads(response.content)
            return evaluation
        except:
            return {
                "is_relevant": "partial",
                "relevant_info": "",
                "relevance_score": 0.5,
                "reasoning": "Unable to evaluate"
            }
    
    def retrieve(self, query: str, k: int = 5) -> Dict:
        """RAFT-style retrieval with relevance evaluation"""
        
        print(f"🎯 RAFT retrieval (evaluating top-{k})")
        
        if not self.vectorstore:
            return {"context": "", "evaluated_docs": [], "num_relevant": 0}
        
        # Get candidate documents
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Evaluate each document
        evaluated_docs = []
        relevant_context = []
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"   Evaluating chunk {i}/{k}...")
            
            # RAFT evaluation
            evaluation = self.evaluate_relevance(query, doc.page_content)
            
            doc_info = {
                "content": doc.page_content,
                "similarity_score": float(score),
                "raft_evaluation": evaluation,
                "rank": i
            }
            
            evaluated_docs.append(doc_info)
            
            # Only include if relevant
            if evaluation["is_relevant"] in ["yes", "partial"]:
                if evaluation["relevance_score"] > 0.5:
                    relevant_context.append(
                        f"[Document {i} - RAFT Score: {evaluation['relevance_score']:.2f}]\n"
                        f"{doc.page_content}\n"
                        f"Relevant Info: {evaluation['relevant_info']}"
                    )
        
        context = "\n\n".join(relevant_context)
        num_relevant = len(relevant_context)
        
        print(f"   {num_relevant}/{k} documents marked as relevant")
        
        return {
            "context": context,
            "evaluated_docs": evaluated_docs,
            "num_relevant": num_relevant
        }


# Test RAFT retriever
if __name__ == "__main__":
    
    docs = [
        {
            "id": "doc_1",
            "content": "KYC requires government ID, proof of address, and SSN. Must complete within 30 days."
        },
        {
            "id": "doc_2",
            "content": "Credit cards offer 2-5% cashback. Annual fees range from $0 to $550."
        },
        {
            "id": "doc_3",
            "content": "AML monitors suspicious transactions. SARs filed within 30 days."
        }
    ]
    
    retriever = RAFTRetriever()
    retriever.build_vectorstore(docs)
    
    result = retriever.retrieve("What documents are needed for KYC?", k=3)
    
    print("\n" + "="*70)
    print("RAFT RETRIEVAL RESULT")
    print("="*70)
    print(result["context"])
    
    print("\n" + "="*70)
    print("RAFT EVALUATIONS")
    print("="*70)
    for doc in result["evaluated_docs"]:
        eval_info = doc["raft_evaluation"]
        print(f"\nDocument {doc['rank']}:")
        print(f"  Relevant: {eval_info['is_relevant']}")
        print(f"  Score: {eval_info['relevance_score']:.2f}")
        print(f"  Reasoning: {eval_info['reasoning']}")