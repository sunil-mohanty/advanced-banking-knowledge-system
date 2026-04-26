"""
RAFT Retriever — Optimized
Key changes:
  - BIGGEST WIN: evaluate_relevance calls now run IN PARALLEL (was sequential)
  - gpt-4o-mini for relevance evaluation (was gpt-4 — overkill for scoring)
  - Shorter evaluation prompt = faster responses
  - Score threshold configurable
  - Vectorstore built once and reused
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import hashlib
from dotenv import load_dotenv

load_dotenv()


class RAFTRetriever:
    def __init__(self):
        # ✅ gpt-4o-mini: perfectly sufficient for binary relevance scoring
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = None
        self._built_hash = None

    def _doc_hash(self, documents: List[Dict]) -> str:
        content = "".join(d.get("content", "") for d in documents)
        return hashlib.md5(content.encode()).hexdigest()

    def build_vectorstore(self, documents: List[Dict]):
        doc_hash = self._doc_hash(documents)

        # ✅ Skip rebuild if same documents
        if self._built_hash == doc_hash and self.vectorstore:
            print("   ⚡ RAFT vectorstore from cache")
            return

        print("🎯 Building RAFT vector store...")

        docs = [Document(
            page_content=doc["content"],
            metadata={"doc_id": doc.get("id", "unknown")}
        ) for doc in documents]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./raft_vectorstore_advanced"
        )

        self._built_hash = doc_hash
        print(f"   RAFT store: {len(chunks)} chunks")

    def evaluate_relevance(self, query: str, document: str) -> Dict:
        # ✅ Much shorter prompt — fewer tokens, faster response
        prompt = f"""Is this document relevant to the query? Return ONLY JSON.

Query: {query}
Document: {document[:600]}

{{"is_relevant":"yes|no|partial","relevance_score":0.0,"relevant_info":"brief extract"}}"""

        response = self.llm.invoke(prompt)

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"```[a-z]*\n?", "", content).strip().rstrip("```")
            return json.loads(content)
        except Exception:
            return {"is_relevant": "partial", "relevance_score": 0.5, "relevant_info": ""}

    def retrieve(self, query: str, k: int = 5, score_threshold: float = 0.5) -> Dict:
        print(f"🎯 RAFT retrieval (parallel evaluation of top-{k})")

        if not self.vectorstore:
            return {"context": "", "evaluated_docs": [], "num_relevant": 0}

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        # ✅ PARALLEL evaluation — was sequential loop, now concurrent
        # This is the single biggest speedup: k=5 calls run simultaneously
        evaluated_docs = [None] * len(results)

        with ThreadPoolExecutor(max_workers=k) as executor:
            future_to_idx = {
                executor.submit(self.evaluate_relevance, query, doc.page_content): i
                for i, (doc, _) in enumerate(results)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                doc, score = results[idx]
                try:
                    evaluation = future.result()
                except Exception:
                    evaluation = {"is_relevant": "partial",
                                  "relevance_score": 0.5, "relevant_info": ""}

                evaluated_docs[idx] = {
                    "content": doc.page_content,
                    "similarity_score": float(score),
                    "raft_evaluation": evaluation,
                    "rank": idx + 1
                }

        # Build context from relevant docs
        relevant_context = []
        for doc_info in evaluated_docs:
            if doc_info is None:
                continue
            ev = doc_info["raft_evaluation"]
            if (ev.get("is_relevant") in ["yes", "partial"] and
                    ev.get("relevance_score", 0) > score_threshold):
                relevant_context.append(
                    f"[Doc {doc_info['rank']} - RAFT: {ev['relevance_score']:.2f}]\n"
                    f"{doc_info['content']}\n"
                    f"Relevant: {ev.get('relevant_info', '')}"
                )

        num_relevant = len(relevant_context)
        print(f"   {num_relevant}/{k} docs relevant")

        return {
            "context": "\n\n".join(relevant_context),
            "evaluated_docs": evaluated_docs,
            "num_relevant": num_relevant
        }
