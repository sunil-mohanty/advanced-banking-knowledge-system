"""
Retrieval Fusion — Optimized
Key changes:
  - Graph + Vector + RAFT retrieval run IN PARALLEL (was sequential)
  - gpt-4o-mini for answer generation on simple/medium queries (was always gpt-4)
  - Answer cache: same query = instant response
  - Build all retrievers in parallel
"""

from langchain_openai import ChatOpenAI
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Ensure src/ is always on path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrievers.graph_retriever import GraphRetriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.raft_retriever import RAFTRetriever
from query_analyzer import QueryAnalyzer
from dotenv import load_dotenv

load_dotenv()


class RetrieverFusion:
    def __init__(self):
        # ✅ Use mini for most answers; only gpt-4o for complex queries
        self.llm_fast   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.llm_strong = ChatOpenAI(model="gpt-4o",      temperature=0)

        self.query_analyzer   = QueryAnalyzer()
        self.graph_retriever  = GraphRetriever()
        self.vector_retriever = VectorRetriever()
        self.raft_retriever   = RAFTRetriever()

        self._answer_cache: Dict[str, Dict] = {}
        print("🔧 Retriever Fusion initialized")

    def build_all_retrievers(self, documents: List[Dict]):
        """Build all retrievers IN PARALLEL — was sequential"""
        print("\n🔨 Building all retrieval systems (parallel)...")
        print("=" * 70)

        # ✅ All three build steps run concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.graph_retriever.build_graph_from_documents, documents): "graph",
                executor.submit(self.vector_retriever.build_vectorstore, documents): "vector",
                executor.submit(self.raft_retriever.build_vectorstore, documents): "raft",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                    print(f"   ✅ {name} retriever ready")
                except Exception as e:
                    print(f"   ⚠️  {name} retriever failed: {e}")

        print("=" * 70)
        print("✅ All retrievers ready!\n")

    def retrieve_all(self, query: str, entities: List[str] = None) -> Dict:
        """Execute all retrievals IN PARALLEL — was sequential"""
        print(f"\n🔍 Parallel multi-retrieval for: '{query}'")

        results = {
            "graph":  {"context": "", "num_nodes": 0},
            "vector": {"context": "", "num_chunks": 0},
            "raft":   {"context": "", "num_relevant": 0},
        }

        # ✅ All three retrievers fire simultaneously
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            if entities:
                futures[executor.submit(self.graph_retriever.retrieve, entities, 2)] = "graph"

            futures[executor.submit(self.vector_retriever.retrieve, query, 5)] = "vector"
            futures[executor.submit(self.raft_retriever.retrieve, query, 5)]   = "raft"

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                    if name == "graph":
                        print(f"   ✓ Graph:  {results[name]['num_nodes']} nodes")
                    elif name == "vector":
                        print(f"   ✓ Vector: {results[name]['num_chunks']} chunks")
                    else:
                        print(f"   ✓ RAFT:   {results[name]['num_relevant']} relevant")
                except Exception as e:
                    print(f"   ⚠️  {name} retrieval failed: {e}")

        return results

    def fuse_contexts(self, retrieval_results: Dict, weights: Dict) -> str:
        parts = []

        if retrieval_results["graph"]["context"] and weights["graph_weight"] > 0:
            parts.append(
                f"**Knowledge Graph ({weights['graph_weight']:.0%}):**\n"
                f"{retrieval_results['graph']['context']}"
            )
        if retrieval_results["vector"]["context"] and weights["vector_weight"] > 0:
            parts.append(
                f"**Semantic Search ({weights['vector_weight']:.0%}):**\n"
                f"{retrieval_results['vector']['context']}"
            )
        if retrieval_results["raft"]["context"] and weights["raft_weight"] > 0:
            parts.append(
                f"**RAFT-Evaluated ({weights['raft_weight']:.0%}):**\n"
                f"{retrieval_results['raft']['context']}"
            )

        return "\n\n".join(parts)

    def adaptive_retrieve(self, query: str) -> Dict:
        # ✅ Full answer cache — repeat queries are instant
        if query in self._answer_cache:
            print("   ⚡ Full answer from cache")
            return self._answer_cache[query]

        print(f"\n{'='*70}")
        print(f"🎯 ADAPTIVE RETRIEVAL")
        print(f"{'='*70}")

        # Step 1 — Analyze query
        analysis = self.query_analyzer.analyze(query)
        weights  = self.query_analyzer.get_retrieval_strategy(analysis)

        # Step 2 — Parallel retrieval
        retrieval_results = self.retrieve_all(query, analysis.entities)

        # Step 3 — Fuse
        fused_context = self.fuse_contexts(retrieval_results, weights)

        # Step 4 — Generate answer
        answer = self._generate_answer(query, fused_context, analysis)

        result = {
            "query": query,
            "analysis": analysis,
            "weights": weights,
            "retrieval_results": retrieval_results,
            "fused_context": fused_context,
            "answer": answer
        }

        self._answer_cache[query] = result
        return result

    def _generate_answer(self, query: str, context: str, analysis) -> str:
        print(f"\n💭 Generating answer...")

        # ✅ Use fast model for simple queries, strong model only for complex
        llm = self.llm_strong if analysis.complexity == "complex" else self.llm_fast

        if analysis.complexity == "simple":
            prompt = f"""Answer this banking question directly and concisely.

Context:
{context}

Question: {query}

Answer:"""

        elif analysis.complexity == "medium":
            prompt = f"""Answer this banking question using the provided context.

Context:
{context}

Question: {query}

Provide a comprehensive answer:"""

        else:  # complex
            prompt = f"""You are a banking expert. Synthesize multiple sources to answer this question.

Multi-source Context:
{context}

Question: {query}

Instructions:
1. Synthesize information across sources
2. Explain your reasoning
3. Provide a well-structured detailed answer

Answer:"""

        response = llm.invoke(prompt)
        print("✅ Answer generated")
        return response.content

    def explain_retrieval_process(self, result: Dict) -> str:
        analysis   = result["analysis"]
        weights    = result["weights"]
        retrieval  = result["retrieval_results"]

        lines = [
            "**Query Analysis:**",
            f"- Intent: {analysis.intent}",
            f"- Complexity: {analysis.complexity}",
            f"- Entities: {', '.join(analysis.entities) or 'none'}",
            "",
            "**Retrieval Strategy:**",
            f"- Graph:  {weights['graph_weight']:.0%}",
            f"- Vector: {weights['vector_weight']:.0%}",
            f"- RAFT:   {weights['raft_weight']:.0%}",
            "",
            "**Sources Retrieved:**",
            f"- Graph entities:  {retrieval['graph']['num_nodes']}",
            f"- Vector chunks:   {retrieval['vector']['num_chunks']}",
            f"- RAFT relevant:   {retrieval['raft']['num_relevant']}",
        ]

        return "\n".join(lines)
