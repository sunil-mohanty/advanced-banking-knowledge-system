"""
Main Advanced Banking Knowledge System
Combines all components into unified interface
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from retriever_fusion import RetrieverFusion
from typing import List, Dict
import time
import json

class AdvancedBankingKnowledgeSystem:
    """
    Complete advanced knowledge system with:
    - GraphRAG for relationships
    - Vector search for semantics
    - RAFT for relevance
    - Intelligent fusion
    """
    
    def __init__(self):
        self.fusion = RetrieverFusion()
        self.query_history = []
        
        print("\n" + "="*70)
        print("🏦 Advanced Banking Knowledge System")
        print("="*70)
    
    def initialize(self, documents: List[Dict]):
        """Initialize all retrieval systems"""
        
        print("\n🚀 Initializing system...")
        start_time = time.time()
        
        self.fusion.build_all_retrievers(documents)
        
        initialization_time = time.time() - start_time
        print(f"\n✅ System initialized in {initialization_time:.2f}s")
    
    def query(self, question: str, return_details: bool = False) -> Dict:
        """
        Query the knowledge system
        
        Args:
            question: User's question
            return_details: If True, return full retrieval details
        
        Returns:
            Dict with answer and optional details
        """
        
        start_time = time.time()
        
        # Execute adaptive retrieval
        result = self.fusion.adaptive_retrieve(question)
        
        query_time = time.time() - start_time
        
        # Store in history
        self.query_history.append({
            "query": question,
            "answer": result["answer"],
            "timestamp": time.time(),
            "query_time": query_time
        })
        
        # Prepare response
        response = {
            "query": question,
            "answer": result["answer"],
            "query_time": query_time
        }
        
        if return_details:
            response["analysis"] = {
                "intent": result["analysis"].intent,
                "complexity": result["analysis"].complexity,
                "entities": result["analysis"].entities
            }
            response["retrieval_strategy"] = result["weights"]
            response["sources"] = {
                "graph_entities": result["retrieval_results"]["graph"]["num_nodes"],
                "vector_chunks": result["retrieval_results"]["vector"]["num_chunks"],
                "raft_relevant": result["retrieval_results"]["raft"]["num_relevant"]
            }
            response["explanation"] = self.fusion.explain_retrieval_process(result)
        
        return response
    
    def batch_query(self, questions: List[str]) -> List[Dict]:
        """Process multiple queries"""
        
        print(f"\n📊 Processing {len(questions)} queries...")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\nQuery {i}/{len(questions)}: {question}")
            result = self.query(question)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get system usage statistics"""
        
        if not self.query_history:
            return {
                "total_queries": 0,
                "average_time": 0,
                "fastest_query": 0,
                "slowest_query": 0
            }
        
        query_times = [q["query_time"] for q in self.query_history]
        
        return {
            "total_queries": len(self.query_history),
            "average_time": sum(query_times) / len(query_times),
            "fastest_query": min(query_times),
            "slowest_query": max(query_times)
        }
    
    def export_history(self, filename: str = "query_history.json"):
        """Export query history to JSON"""
        
        with open(filename, 'w') as f:
            json.dump(self.query_history, f, indent=2)
        
        print(f"💾 Exported {len(self.query_history)} queries to {filename}")


# Test the complete system
if __name__ == "__main__":
    
    # Banking knowledge base
    banking_kb = [
        {
            "id": "kyc_policy",
            "content": """Know Your Customer (KYC) Policy
            
            KYC is enforced by FinCEN under the Bank Secrecy Act (BSA).
            
            Required documentation:
            - Government-issued photo ID
            - Proof of address (< 3 months old)
            - Social Security Number or Tax ID
            - Employment information
            
            Verification timeline: 30 days from account opening
            Penalties for non-compliance: Up to $250,000 per violation
            
            KYC prevents money laundering and terrorist financing."""
        },
        {
            "id": "aml_procedures",
            "content": """Anti-Money Laundering (AML) Procedures
            
            AML programs include:
            - Transaction monitoring systems
            - Suspicious Activity Reports (SARs) to FinCEN
            - Currency Transaction Reports (CTRs) for cash >$10,000
            - OFAC sanctions screening
            
            Red flags:
            - Structuring transactions
            - Wire transfers to high-risk jurisdictions
            - Unusual activity patterns
            
            SAR filing deadline: 30 days from detection
            Annual AML training: Mandatory for all staff"""
        },
        {
            "id": "mortgage_requirements",
            "content": """Mortgage Loan Approval Criteria
            
            Credit requirements:
            - Conventional: 620+ credit score
            - FHA loans: 580+ credit score
            - VA loans: No minimum (case-by-case)
            
            Financial requirements:
            - Debt-to-income ratio: <43% (maximum)
            - Down payment: 3-20% (varies by loan type)
            - Stable income: 2+ years employment history
            
            Property requirements:
            - Appraisal required
            - Homeowners insurance mandatory
            - Property inspection recommended
            
            Interest rates influenced by Federal Reserve policy.
            Fannie Mae and Freddie Mac purchase conforming loans."""
        },
        {
            "id": "cdd_procedures",
            "content": """Customer Due Diligence (CDD)
            
            CDD is part of KYC and involves:
            
            Standard CDD:
            - Customer identification
            - Beneficial ownership identification
            - Understanding nature of customer relationship
            - Ongoing monitoring
            
            Enhanced Due Diligence (EDD) for:
            - Politically Exposed Persons (PEPs)
            - High-risk customers
            - Customers from high-risk jurisdictions
            - Transactions >$100,000
            
            CDD mandated by PATRIOT Act.
            Risk-based approach: Assess customer risk profile."""
        },
        {
            "id": "account_types",
            "content": """Banking Account Types and Features
            
            1. Savings Account
               - Interest rate: 2.5% APY
               - Minimum balance: $100
               - Monthly fee: $0 (with minimum balance)
               - Withdrawal limit: 6 per month
            
            2. Checking Account
               - Interest rate: 0.5% APY
               - Minimum balance: $0
               - Monthly fee: $10 (waived with $500 balance or direct deposit)
               - Unlimited transactions
               - Debit card included
            
            3. Money Market Account
               - Interest rate: 3.5% APY
               - Minimum balance: $2,500
               - Monthly fee: $15 (waived with balance)
               - Limited withdrawals: 6 per month
               - Check writing allowed
            
            4. Certificate of Deposit (CD)
               - Interest rate: 4.0-5.5% (term-based)
               - Terms: 3, 6, 12, 24, 36, 60 months
               - Early withdrawal penalty: 3-6 months interest
               - FDIC insured up to $250,000"""
        }
    ]
    
    # Initialize system
    system = AdvancedBankingKnowledgeSystem()
    system.initialize(banking_kb)
    
    # Test queries
    print("\n\n" + "="*70)
    print("TESTING ADVANCED KNOWLEDGE SYSTEM")
    print("="*70)
    
    test_queries = [
        "What is KYC?",
        "How are KYC and AML related?",
        "What are mortgage requirements?",
        "Explain Customer Due Diligence and when Enhanced Due Diligence is needed"
    ]
    
    for query in test_queries:
        print(f"\n{'#'*70}")
        print(f"Query: {query}")
        print(f"{'#'*70}")
        
        result = system.query(query, return_details=True)
        
        print(f"\n📊 Query Analysis:")
        print(f"   Intent: {result['analysis']['intent']}")
        print(f"   Complexity: {result['analysis']['complexity']}")
        print(f"   Time: {result['query_time']:.2f}s")
        
        print(f"\n📚 Sources Used:")
        print(f"   Graph entities: {result['sources']['graph_entities']}")
        print(f"   Vector chunks: {result['sources']['vector_chunks']}")
        print(f"   RAFT relevant: {result['sources']['raft_relevant']}")
        
        print(f"\n✨ Answer:")
        print(f"{result['answer']}")
    
    # Show statistics
    print(f"\n\n{'='*70}")
    print("SYSTEM STATISTICS")
    print(f"{'='*70}")
    
    stats = system.get_statistics()
    print(f"Total queries: {stats['total_queries']}")
    print(f"Average time: {stats['average_time']:.2f}s")
    print(f"Fastest: {stats['fastest_query']:.2f}s")
    print(f"Slowest: {stats['slowest_query']:.2f}s")
    
    # Export history
    system.export_history("advanced_kb_test_history.json")