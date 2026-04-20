"""
Query Analyzer
Analyzes queries to determine optimal retrieval strategy
"""

from langchain_openai import ChatOpenAI
from typing import Dict, List
from pydantic import BaseModel
import json
from dotenv import load_dotenv

load_dotenv()

class QueryAnalysis(BaseModel):
    """Structured query analysis"""
    intent: str  # factual, comparative, procedural, analytical
    complexity: str  # simple, medium, complex
    entities: List[str]
    requires_graph: bool
    requires_vector: bool
    requires_multi_hop: bool
    reasoning_depth: str  # shallow, medium, deep
    
class QueryAnalyzer:
    """Analyze queries to determine optimal retrieval strategy"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze query and determine retrieval strategy"""
        
        prompt = f"""Analyze this banking query and determine the best retrieval strategy.

Query: {query}

Analyze:
1. **Intent**: What is the user trying to do?
   - factual: Looking for specific facts
   - comparative: Comparing options/approaches
   - procedural: Understanding a process/workflow
   - analytical: Requiring analysis/calculation

2. **Complexity**: How complex is the query?
   - simple: Single fact lookup
   - medium: Multiple facts or light reasoning
   - complex: Multi-hop reasoning or synthesis

3. **Entities**: What key entities are mentioned?
   - List specific terms, regulations, products, etc.

4. **Retrieval Needs**:
   - requires_graph: Would graph relationships help? (e.g., "how are X and Y related?")
   - requires_vector: Would semantic search help? (e.g., finding similar concepts)
   - requires_multi_hop: Requires connecting multiple pieces of info?

5. **Reasoning Depth**:
   - shallow: Direct answer from single source
   - medium: Combining 2-3 sources
   - deep: Complex reasoning across multiple sources

Respond in JSON:
{{
    "intent": "factual/comparative/procedural/analytical",
    "complexity": "simple/medium/complex",
    "entities": ["entity1", "entity2"],
    "requires_graph": true/false,
    "requires_vector": true/false,
    "requires_multi_hop": true/false,
    "reasoning_depth": "shallow/medium/deep"
}}

JSON:"""
        
        response = self.llm.invoke(prompt)
        
        try:
            analysis_dict = json.loads(response.content)
            analysis = QueryAnalysis(**analysis_dict)
            
            print(f"\n🔍 Query Analysis:")
            print(f"   Intent: {analysis.intent}")
            print(f"   Complexity: {analysis.complexity}")
            print(f"   Entities: {', '.join(analysis.entities)}")
            print(f"   Graph needed: {analysis.requires_graph}")
            print(f"   Vector needed: {analysis.requires_vector}")
            print(f"   Multi-hop: {analysis.requires_multi_hop}")
            
            return analysis
            
        except Exception as e:
            print(f"⚠️  Analysis parsing failed: {e}")
            # Fallback to conservative defaults
            return QueryAnalysis(
                intent="factual",
                complexity="medium",
                entities=[],
                requires_graph=True,
                requires_vector=True,
                requires_multi_hop=False,
                reasoning_depth="medium"
            )
    
    def get_retrieval_strategy(self, analysis: QueryAnalysis) -> Dict:
        """Determine retrieval weights based on analysis"""
        
        # Default weights
        weights = {
            "graph_weight": 0.33,
            "vector_weight": 0.33,
            "raft_weight": 0.34
        }
        
        # Adjust based on analysis
        if analysis.requires_graph and analysis.requires_multi_hop:
            # Graph is very important for multi-hop
            weights["graph_weight"] = 0.5
            weights["vector_weight"] = 0.25
            weights["raft_weight"] = 0.25
            
        elif analysis.intent == "comparative":
            # Vector search good for comparisons
            weights["graph_weight"] = 0.2
            weights["vector_weight"] = 0.5
            weights["raft_weight"] = 0.3
            
        elif analysis.complexity == "simple":
            # RAFT good for simple factual queries
            weights["graph_weight"] = 0.2
            weights["vector_weight"] = 0.3
            weights["raft_weight"] = 0.5
        
        print(f"\n⚖️  Retrieval Weights:")
        print(f"   Graph: {weights['graph_weight']:.2%}")
        print(f"   Vector: {weights['vector_weight']:.2%}")
        print(f"   RAFT: {weights['raft_weight']:.2%}")
        
        return weights


# Test the analyzer
if __name__ == "__main__":
    
    analyzer = QueryAnalyzer()
    
    test_queries = [
        "What is KYC?",
        "How are KYC and AML related?",
        "Compare mortgage loans and personal loans",
        "Explain the complete process of opening a business account"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        analysis = analyzer.analyze(query)
        strategy = analyzer.get_retrieval_strategy(analysis)