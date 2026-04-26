"""
Graph Retriever — Optimized
Key changes:
  - Parallel document extraction via ThreadPoolExecutor (was sequential)
  - gpt-4o-mini for entity extraction (was gpt-4)
  - Graph built once, cached — never rebuilt for same documents
  - Tighter JSON prompt reduces token usage
"""

from langchain_openai import ChatOpenAI
import networkx as nx
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import hashlib
from dotenv import load_dotenv

load_dotenv()


class GraphRetriever:
    def __init__(self):
        # ✅ gpt-4o-mini for extraction: sufficient quality, much faster
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.graph = nx.DiGraph()
        self._built_hash = None  # Track if graph needs rebuild

    def _doc_hash(self, documents: List[Dict]) -> str:
        content = "".join(d.get("content", "") for d in documents)
        return hashlib.md5(content.encode()).hexdigest()

    def build_graph_from_documents(self, documents: List[Dict]):
        doc_hash = self._doc_hash(documents)

        # ✅ Skip rebuild if documents unchanged
        if self._built_hash == doc_hash and self.graph.number_of_nodes() > 0:
            print(f"   ⚡ Graph from cache ({self.graph.number_of_nodes()} nodes)")
            return

        print("🕸️  Building knowledge graph (parallel)...")
        self.graph.clear()

        # ✅ Extract all docs IN PARALLEL — was sequential, now concurrent
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    self._extract_entities_relationships,
                    doc["content"],
                    doc.get("id", f"doc_{i}")
                ): i
                for i, doc in enumerate(documents, 1)
            }

            for future in as_completed(futures):
                try:
                    extraction = future.result()
                    self._add_to_graph(extraction)
                except Exception as e:
                    print(f"   ⚠️  Extraction error: {e}")

        self._built_hash = doc_hash
        print(f"   Graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

    def _extract_entities_relationships(self, text: str, doc_id: str) -> Dict:
        # ✅ Shorter prompt = fewer tokens = faster + cheaper
        prompt = f"""Extract banking entities and relationships. Return ONLY JSON, no markdown.

Text: {text[:1500]}

{{"entities":[{{"name":"KYC","type":"regulation"}}],
  "relationships":[{{"source":"KYC","target":"FinCEN","relation":"regulated_by"}}]}}"""

        response = self.llm.invoke(prompt)

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"```[a-z]*\n?", "", content).strip().rstrip("```")
            result = json.loads(content)
            for entity in result.get("entities", []):
                entity["doc_id"] = doc_id
            return result
        except Exception:
            return {"entities": [], "relationships": []}

    def _add_to_graph(self, extraction: Dict):
        for entity in extraction.get("entities", []):
            name = entity["name"]
            if name not in self.graph:
                self.graph.add_node(
                    name,
                    entity_type=entity.get("type", "concept"),
                    doc_ids=[entity.get("doc_id")]
                )
            else:
                self.graph.nodes[name]["doc_ids"].append(entity.get("doc_id"))

        for rel in extraction.get("relationships", []):
            source, target = rel.get("source"), rel.get("target")
            if source and target:
                self.graph.add_edge(source, target,
                                    relation=rel.get("relation", "relates_to"))

    def retrieve(self, entities: List[str], max_hops: int = 2) -> Dict:
        relevant_nodes: Set[str] = set()
        paths = []

        for entity in entities:
            # ✅ Fuzzy match — find entity even if case differs
            matched = next(
                (n for n in self.graph.nodes if n.lower() == entity.lower()), None
            )
            if not matched:
                continue

            relevant_nodes.add(matched)
            relevant_nodes.update(self._bfs_neighbors(matched, max_hops))

            for other in entities:
                other_matched = next(
                    (n for n in self.graph.nodes if n.lower() == other.lower()), None
                )
                if other_matched and other_matched != matched:
                    try:
                        path = nx.shortest_path(self.graph, matched, other_matched)
                        if len(path) > 1:
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        pass

        context = self._build_context(relevant_nodes, paths)

        return {
            "context": context,
            "nodes": list(relevant_nodes),
            "paths": paths,
            "num_nodes": len(relevant_nodes)
        }

    def _bfs_neighbors(self, entity: str, max_hops: int) -> Set[str]:
        neighbors: Set[str] = set()
        current_level = {entity}

        for _ in range(max_hops):
            next_level: Set[str] = set()
            for node in current_level:
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            neighbors.update(next_level)
            current_level = next_level

        return neighbors

    def _build_context(self, nodes: Set[str], paths: List) -> str:
        parts = ["**Entity Relationships:**"]

        for node in nodes:
            node_info = [f"\n{node}:"]
            for succ in self.graph.successors(node):
                if succ in nodes:
                    rel = self.graph.get_edge_data(node, succ).get("relation", "relates_to")
                    node_info.append(f"  • {rel} → {succ}")
            for pred in self.graph.predecessors(node):
                if pred in nodes:
                    rel = self.graph.get_edge_data(pred, node).get("relation", "relates_to")
                    node_info.append(f"  • {pred} → {rel}")
            if len(node_info) > 1:
                parts.append("\n".join(node_info))

        if paths:
            parts.append("\n**Connection Paths:**")
            for path in paths:
                parts.append(f"  • {' → '.join(path)}")

        return "\n".join(parts)
