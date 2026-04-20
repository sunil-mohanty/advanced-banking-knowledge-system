"""
Intelligent Retrieval Fusion
Combines Graph, Vector, and RAFT retrievers with weighted fusion
"""

from langchain_openai import ChatOpenAI
from typing import Dict, List
from retrievers.graph_retriever import GraphRetriever
from retrievers.vector_retriever import VectorRetriever
from retrievers.raft_retriever import RAFTRetriever
from query_analyzer import QueryAnalyzer
from dotenv import load_dotenv

load_dotenv()

class RetrieverFusion:
    """Intelligently fuse multiple retrieval strategies"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.query_analyzer = QueryAnalyzer()
        
        # Initialize retrievers
        self.graph_retriever = GraphRetriever()
        self.vector_retriever = VectorRetriever()
        self.raft_retriever = RAFTRetriever()
        
        print("🔧 Retriever Fusion initialized")
    
    def build_all_retrievers(self, documents: List[Dict]):
        """Build all retrieval systems from documents"""
        
        print("\n🔨 Building all retrieval systems...")
        print("="*70)
        
        # Build graph
        self.graph_retriever.build_graph_from_documents(documents)
        
        # Build vector store
        self.vector_retriever.build_vectorstore(documents)
        
        # Build RAFT store
        self.raft_retriever.build_vectorstore(documents)
        
        print("="*70)
        print("✅ All retrievers ready!\n")
    
    def retrieve_all(self, query: str, entities: List[str] = None) -> Dict:
        """Execute all retrieval strategies"""
        
        print(f"\n{'='*70}")
        print(f"🔍 MULTI-RETRIEVAL for: '{query}'")
        print(f"{'='*70}\n")
        
        results = {}
        
        # Graph retrieval
        if entities:
            try:
                graph_result = self.graph_retriever.retrieve(entities, max_hops=2)
                results["graph"] = graph_result
                print(f"✓ Graph: {graph_result['num_nodes']} nodes")
            except Exception as e:
                print(f"⚠️  Graph retrieval failed: {e}")
                results["graph"] = {"context": "", "num_nodes": 0}
        else:
            results["graph"] = {"context": "", "num_nodes": 0}
        
        # Vector retrieval
        try:
            vector_result = self.vector_retriever.retrieve(query, k=5)
            results["vector"] = vector_result
            print(f"✓ Vector: {vector_result['num_chunks']} chunks")
        except Exception as e:
            print(f"⚠️  Vector retrieval failed: {e}")
            results["vector"] = {"context": "", "num_chunks": 0}
        
        # RAFT retrieval
        try:
            raft_result = self.raft_retriever.retrieve(query, k=5)
            results["raft"] = raft_result
            print(f"✓ RAFT: {raft_result['num_relevant']} relevant docs")
        except Exception as e:
            print(f"⚠️  RAFT retrieval failed: {e}")
            results["raft"] = {"context": "", "num_relevant": 0}
        
        return results
    
    def fuse_contexts(
        self,
        retrieval_results: Dict,
        weights: Dict
    ) -> str:
        """Fuse contexts from different retrievers with weights"""
        
        print(f"\n🎨 Fusing contexts with weights:")
        print(f"   Graph: {weights['graph_weight']:.2%}")
        print(f"   Vector: {weights['vector_weight']:.2%}")
        print(f"   RAFT: {weights['raft_weight']:.2%}")
        
        fused_parts = []
        
        # Add graph context
        if retrieval_results["graph"]["context"] and weights["graph_weight"] > 0:
            fused_parts.append(
                f"**Knowledge Graph Context (weight: {weights['graph_weight']:.2%}):**\n"
                f"{retrieval_results['graph']['context']}"
            )
        
        # Add vector context
        if retrieval_results["vector"]["context"] and weights["vector_weight"] > 0:
            fused_parts.append(
                f"\n**Semantic Search Context (weight: {weights['vector_weight']:.2%}):**\n"
                f"{retrieval_results['vector']['context']}"
            )
        
        # Add RAFT context
        if retrieval_results["raft"]["context"] and weights["raft_weight"] > 0:
            fused_parts.append(
                f"\n**RAFT-Evaluated Context (weight: {weights['raft_weight']:.2%}):**\n"
                f"{retrieval_results['raft']['context']}"
            )
        
        return "\n\n".join(fused_parts)
    
    def adaptive_retrieve(self, query: str) -> Dict:
        """
        Adaptive retrieval with automatic strategy selection
        
        1. Analyze query
        2. Execute appropriate retrievers
        3. Fuse results intelligently
        4. Generate answer
        """
        
        print(f"\n{'='*70}")
        print(f"🎯 ADAPTIVE RETRIEVAL")
        print(f"{'='*70}")
        
        # Step 1: Analyze query
        analysis = self.query_analyzer.analyze(query)
        weights = self.query_analyzer.get_retrieval_strategy(analysis)
        
        # Step 2: Execute retrievals
        retrieval_results = self.retrieve_all(query, analysis.entities)
        
        # Step 3: Fuse contexts
        fused_context = self.fuse_contexts(retrieval_results, weights)
        
        # Step 4: Generate answer
        answer = self._generate_answer(query, fused_context, analysis)
        
        return {
            "query": query,
            "analysis": analysis,
            "weights": weights,
            "retrieval_results": retrieval_results,
            "fused_context": fused_context,
            "answer": answer
        }
    
    def _generate_answer(
        self,
        query: str,
        context: str,
        analysis
    ) -> str:
        """Generate final answer using fused context"""
        
        print(f"\n💭 Generating answer...")
        
        # Create prompt based on query complexity
        if analysis.complexity == "simple":
            prompt_template = """Answer this banking question directly and concisely.

Context:
{context}

Question: {query}

Provide a clear, factual answer:"""
        
        elif analysis.complexity == "medium":
            prompt_template = """Answer this banking question using the provided context from multiple sources.

Context from multiple retrieval methods:
{context}

Question: {query}

Provide a comprehensive answer:"""
        
        else:  # complex
            prompt_template = """You are a banking expert. Answer this complex question by reasoning through multiple sources.

Multi-source Context:
{context}

Question: {query}

**Instructions:**
1. Synthesize information from different sources
2. Explain your reasoning
3. Provide a detailed, well-structured answer

Answer:"""
        
        prompt = prompt_template.format(context=context, query=query)
        response = self.llm.invoke(prompt)
        
        print(f"✅ Answer generated")
        
        return response.content
    
    def explain_retrieval_process(self, result: Dict) -> str:
        """Generate human-readable explanation of retrieval process"""
        
        explanation_parts = []
        
        # Query analysis
        analysis = result["analysis"]
        explanation_parts.append(f"**Query Analysis:**")
        explanation_parts.append(f"- Intent: {analysis.intent}")
        explanation_parts.append(f"- Complexity: {analysis.complexity}")
        explanation_parts.append(f"- Key entities: {', '.join(analysis.entities) if analysis.entities else 'None detected'}")
        
        # Retrieval strategy
        weights = result["weights"]
        explanation_parts.append(f"\n**Retrieval Strategy:**")
        explanation_parts.append(f"- Graph retrieval: {weights['graph_weight']:.0%}")
        explanation_parts.append(f"- Vector search: {weights['vector_weight']:.0%}")
        explanation_parts.append(f"- RAFT evaluation: {weights['raft_weight']:.0%}")
        
        # Results summary
        retrieval = result["retrieval_results"]
        explanation_parts.append(f"\n**Sources Retrieved:**")
        explanation_parts.append(f"- Knowledge graph: {retrieval['graph']['num_nodes']} entities")
        explanation_parts.append(f"- Semantic search: {retrieval['vector']['num_chunks']} chunks")
        explanation_parts.append(f"- RAFT relevant: {retrieval['raft']['num_relevant']} documents")
        
        return "\n".join(explanation_parts)


# Test the fusion system
if __name__ == "__main__":
    
    # Sample banking corpus
    documents = [
        {
            "id": "kyc_doc",
            "content": """Know Your Customer (KYC) Compliance
            
            KYC is a regulatory requirement enforced by FinCEN under the Bank Secrecy Act.
            Financial institutions must verify customer identity using:
            - Government-issued photo ID (passport, driver's license)
            - Proof of address (utility bill, bank statement)
            - Social Security Number or Tax ID
            
            Verification must be completed within 30 days of account opening.
            Non-compliance penalties can reach $250,000 per violation."""
        },
        {
            "id": "aml_doc",
            "content": """Anti-Money Laundering (AML) Procedures
            
            AML programs monitor transactions for suspicious activity.
            Banks must file Suspicious Activity Reports (SARs) with FinCEN within 30 days.
            
            Red flags include:
            - Structuring (multiple transactions under $10,000)
            - Wire transfers to high-risk countries
            - Patterns inconsistent with customer profile
            
            Currency Transaction Reports (CTRs) required for cash over $10,000."""
        },
        {
            "id": "mortgage_doc",
            "content": """Mortgage Loan Requirements
            
            Mortgage approval criteria:
            - Minimum credit score: 620 (conventional), 580 (FHA)
            - Debt-to-income ratio: Maximum 43%
            - Down payment: 3-20% depending on loan type
            - Employment: 2+ years stable employment history
            
            Interest rates affected by Federal Reserve policy.
            Fannie Mae and Freddie Mac purchase conforming loans."""
        },
        {
            "id": "account_types_doc",
            "content": """Banking Account Types
            
            1. Savings Account
               - Interest: 2.5% APY
               - Minimum balance: $100
               - Monthly fee: $0 (if balance maintained)
            
            2. Checking Account
               - Interest: 0.5% APY
               - Monthly fee: $10 (waived with $500 balance)
               - Unlimited transactions
            
            3. Money Market Account
               - Interest: 3.5% APY
               - Minimum balance: $2,500
               - Limited withdrawals: 6 per month"""
        },
        {
            "id": "cdd_doc",
            "content": """Customer Due Diligence (CDD)
            
            CDD is part of KYC compliance and involves:
            - Risk assessment based on customer profile
            - Enhanced Due Diligence (EDD) for high-risk customers
            - Politically Exposed Persons (PEPs) require additional screening
            
            CDD is required under the PATRIOT Act.
            Ongoing monitoring ensures continued compliance."""
        }
    ]
    
    # Initialize fusion system
    fusion = RetrieverFusion()
    fusion.build_all_retrievers(documents)
    
    # Test queries
    test_queries = [
        "What is KYC?",
        "How are KYC and AML related?",
        "What are the requirements for getting a mortgage loan?",
        "Explain the complete customer verification process including ongoing monitoring"
    ]
    
    for query in test_queries:
        print(f"\n\n{'#'*70}")
        print(f"# TESTING: {query}")
        print(f"{'#'*70}\n")
        
        result = fusion.adaptive_retrieve(query)
        
        # Display results
        print(f"\n{'='*70}")
        print(f"RETRIEVAL PROCESS EXPLANATION")
        print(f"{'='*70}")
        print(fusion.explain_retrieval_process(result))
        
        print(f"\n{'='*70}")
        print(f"FINAL ANSWER")
        print(f"{'='*70}")
        print(result["answer"])
        print(f"{'='*70}")