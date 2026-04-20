"""
Graph-based Retriever using Knowledge Graph
"""

from langchain_openai import ChatOpenAI
import networkx as nx
from typing import List, Dict, Set, Tuple
import json
from dotenv import load_dotenv

load_dotenv()

class GraphRetriever:
    """Retrieve information using knowledge graph traversal"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.graph = nx.DiGraph()
        
    def build_graph_from_documents(self, documents: List[Dict]):
        """Build knowledge graph from documents"""
        
        print("🕸️  Building knowledge graph...")
        
        for i, doc in enumerate(documents, 1):
            # Extract entities and relationships
            extraction = self._extract_entities_relationships(
                doc["content"],
                doc.get("id", f"doc_{i}")
            )
            
            # Add to graph
            self._add_to_graph(extraction)
        
        print(f"   Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _extract_entities_relationships(self, text: str, doc_id: str) -> Dict:
        """Extract entities and relationships from text"""
        
        prompt = f"""Extract entities and relationships from this banking text.

Text: {text}

Extract:
- Entities: regulations, organizations, concepts, products, requirements
- Relationships: how entities connect (regulates, requires, affects, etc.)

Return JSON:
{{
    "entities": [
        {{"name": "KYC", "type": "regulation"}}
    ],
    "relationships": [
        {{"source": "KYC", "target": "FinCEN", "relation": "regulated_by"}}
    ]
}}

JSON:"""
        
        response = self.llm.invoke(prompt)
        
        try:
            result = json.loads(response.content)
            
            # Add doc_id to entities
            for entity in result.get("entities", []):
                entity["doc_id"] = doc_id
            
            return result
        except:
            return {"entities": [], "relationships": []}
    
    def _add_to_graph(self, extraction: Dict):
        """Add entities and relationships to graph"""
        
        # Add entities
        for entity in extraction.get("entities", []):
            name = entity["name"]
            if name not in self.graph:
                self.graph.add_node(
                    name,
                    entity_type=entity.get("type", "concept"),
                    doc_ids=[entity.get("doc_id")]
                )
            else:
                # Update doc_ids
                self.graph.nodes[name]["doc_ids"].append(entity.get("doc_id"))
        
        # Add relationships
        for rel in extraction.get("relationships", []):
            source = rel.get("source")
            target = rel.get("target")
            
            if source and target:
                self.graph.add_edge(
                    source,
                    target,
                    relation=rel.get("relation", "relates_to")
                )
    
    def retrieve(self, entities: List[str], max_hops: int = 2) -> Dict:
        """Retrieve context using graph traversal"""
        
        print(f"🕸️  Graph retrieval for entities: {', '.join(entities)}")
        
        # Collect all relevant nodes
        relevant_nodes = set()
        paths = []
        
        for entity in entities:
            if entity in self.graph:
                # Add entity itself
                relevant_nodes.add(entity)
                
                # BFS to find neighbors
                neighbors = self._bfs_neighbors(entity, max_hops)
                relevant_nodes.update(neighbors)
                
                # Find paths between query entities
                for other_entity in entities:
                    if other_entity != entity and other_entity in self.graph:
                        try:
                            path = nx.shortest_path(self.graph, entity, other_entity)
                            if len(path) > 1:  # Only if path exists
                                paths.append(path)
                        except nx.NetworkXNoPath:
                            pass
        
        # Build context from graph
        context = self._build_context(relevant_nodes, paths)
        
        print(f"   Found {len(relevant_nodes)} relevant nodes")
        print(f"   Found {len(paths)} connection paths")
        
        return {
            "context": context,
            "nodes": list(relevant_nodes),
            "paths": paths,
            "num_nodes": len(relevant_nodes)
        }
    
    def _bfs_neighbors(self, entity: str, max_hops: int) -> Set[str]:
        """BFS to find neighbors within max_hops"""
        
        neighbors = set()
        current_level = {entity}
        
        for _ in range(max_hops):
            next_level = set()
            
            for node in current_level:
                # Successors (outgoing edges)
                next_level.update(self.graph.successors(node))
                # Predecessors (incoming edges)
                next_level.update(self.graph.predecessors(node))
            
            neighbors.update(next_level)
            current_level = next_level
        
        return neighbors
    
    def _build_context(self, nodes: Set[str], paths: List[List[str]]) -> str:
        """Build textual context from graph structure"""
        
        context_parts = []
        
        # Add node information
        context_parts.append("**Entity Relationships:**")
        
        for node in nodes:
            node_info = [f"\n{node}:"]
            
            # Outgoing edges
            for successor in self.graph.successors(node):
                if successor in nodes:
                    edge_data = self.graph.get_edge_data(node, successor)
                    relation = edge_data.get("relation", "relates_to")
                    node_info.append(f"  • {relation} → {successor}")
            
            # Incoming edges
            for predecessor in self.graph.predecessors(node):
                if predecessor in nodes:
                    edge_data = self.graph.get_edge_data(predecessor, node)
                    relation = edge_data.get("relation", "relates_to")
                    node_info.append(f"  • {predecessor} → {relation}")
            
            if len(node_info) > 1:  # Only add if has relationships
                context_parts.append("\n".join(node_info))
        
        # Add paths
        if paths:
            context_parts.append("\n**Connection Paths:**")
            for path in paths:
                path_str = " → ".join(path)
                context_parts.append(f"  • {path_str}")
        
        return "\n".join(context_parts)
    
    def get_subgraph_summary(self, entities: List[str]) -> str:
        """Generate LLM summary of subgraph"""
        
        retrieval = self.retrieve(entities, max_hops=2)
        
        prompt = f"""Summarize the relationships in this knowledge graph subgraph.

Graph Information:
{retrieval['context']}

Provide a 2-3 sentence summary explaining how these concepts are related.

Summary:"""
        
        response = self.llm.invoke(prompt)
        return response.content


# Test graph retriever
if __name__ == "__main__":
    
    # Sample documents
    docs = [
        {
            "id": "doc_1",
            "content": "KYC (Know Your Customer) is regulated by FinCEN under the Bank Secrecy Act. It requires identity verification."
        },
        {
            "id": "doc_2",
            "content": "AML (Anti-Money Laundering) programs monitor transactions. SARs are filed with FinCEN for suspicious activity."
        },
        {
            "id": "doc_3",
            "content": "The Federal Reserve regulates interest rates. Mortgage rates are affected by the federal funds rate."
        }
    ]
    
    # Build and test
    retriever = GraphRetriever()
    retriever.build_graph_from_documents(docs)
    
    # Test retrieval
    result = retriever.retrieve(["KYC", "FinCEN"])
    
    print("\n" + "="*70)
    print("GRAPH RETRIEVAL RESULT")
    print("="*70)
    print(result["context"])
    
    # Test summary
    summary = retriever.get_subgraph_summary(["KYC", "AML", "FinCEN"])
    print("\n" + "="*70)
    print("SUBGRAPH SUMMARY")
    print("="*70)
    print(summary)