"""
Advanced Banking Knowledge System - Streamlit UI
Professional interface with real-time retrieval visualization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src'))

import streamlit as st
from src.knowledge_system import AdvancedBankingKnowledgeSystem
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import time

# Page configuration
st.set_page_config(
    page_title="Advanced Banking Knowledge System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .retriever-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .answer-box {
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
    st.session_state.initialized = False
    st.session_state.query_history = []

# Banking knowledge base
BANKING_KB = [
    {
        "id": "kyc_policy",
        "content": """Know Your Customer (KYC) Compliance Policy
        
        KYC is a regulatory requirement enforced by the Financial Crimes Enforcement Network (FinCEN) 
        under the Bank Secrecy Act (BSA). Financial institutions must verify customer identity before 
        opening accounts.
        
        Required Documentation:
        - Government-issued photo ID (passport, driver's license)
        - Proof of address (utility bill, bank statement less than 3 months old)
        - Social Security Number or Tax Identification Number
        - Employment information and source of funds
        
        Verification Timeline: Must be completed within 30 days of account opening
        
        Non-Compliance Penalties: Fines up to $250,000 per violation and potential criminal charges
        
        Purpose: KYC compliance prevents money laundering, terrorist financing, and identity fraud."""
    },
    {
        "id": "aml_procedures",
        "content": """Anti-Money Laundering (AML) Procedures
        
        Banks must implement comprehensive AML programs to detect and prevent money laundering.
        
        Core Components:
        - Transaction Monitoring: Real-time automated systems monitoring all transactions
        - Suspicious Activity Reports (SARs): Filed with FinCEN within 30 days of detection
        - Currency Transaction Reports (CTRs): Required for cash transactions exceeding $10,000
        - OFAC Screening: Office of Foreign Assets Control sanctions list checking
        
        Red Flags for Suspicious Activity:
        - Structuring: Multiple transactions just below $10,000 threshold
        - Wire transfers to high-risk jurisdictions
        - Transaction patterns inconsistent with customer profile
        - Reluctance to provide required information
        - Unusual cash deposits or withdrawals
        
        Compliance Requirements:
        - Annual AML training for all bank employees
        - Independent audit of AML program annually
        - Designated AML Compliance Officer
        - Risk-based customer due diligence
        
        Penalties: Violations can result in fines up to $500,000 per incident and criminal prosecution."""
    },
    {
        "id": "mortgage_requirements",
        "content": """Mortgage Loan Approval Requirements
        
        Credit Score Requirements:
        - Conventional loans: Minimum 620 credit score
        - FHA loans: Minimum 580 credit score (500-579 requires 10% down)
        - VA loans: No strict minimum (evaluated case-by-case)
        - Jumbo loans: Minimum 700 credit score
        
        Financial Requirements:
        - Debt-to-Income Ratio: Maximum 43% (some lenders allow up to 50%)
        - Down Payment: 3-20% depending on loan type
          * Conventional: 5-20%
          * FHA: 3.5% (with 580+ credit score)
          * VA: 0% for qualified veterans
          * USDA: 0% for eligible rural properties
        - Employment History: Minimum 2 years in same field
        - Income Verification: Last 2 years tax returns, recent pay stubs
        
        Property Requirements:
        - Professional appraisal required
        - Homeowners insurance mandatory
        - Property inspection recommended
        - Title insurance required
        
        Interest Rate Factors:
        - Federal Reserve monetary policy
        - Credit score (better score = lower rate)
        - Loan-to-value ratio
        - Loan term (15-year vs 30-year)
        - Market conditions
        
        Additional Information:
        - Pre-approval valid for 60-90 days
        - Closing costs: 2-5% of loan amount
        - Fannie Mae and Freddie Mac purchase conforming loans
        - Private Mortgage Insurance (PMI) required if down payment <20%"""
    },
    {
        "id": "cdd_procedures",
        "content": """Customer Due Diligence (CDD) Procedures
        
        CDD is a critical component of KYC compliance mandated by the PATRIOT Act and FinCEN regulations.
        
        Standard Customer Due Diligence:
        1. Customer Identification Program (CIP)
           - Verify customer identity
           - Collect identifying information
           - Document verification process
        
        2. Beneficial Ownership Identification
           - Identify individuals owning 25%+ of legal entity
           - Required for business accounts
           - Ultimate beneficial owner (UBO) documentation
        
        3. Understanding Customer Relationships
           - Purpose of account
           - Expected transaction activity
           - Source of funds
           - Nature of business
        
        4. Ongoing Monitoring
           - Regular review of customer activity
           - Update customer information periodically
           - Monitor for suspicious patterns
        
        Enhanced Due Diligence (EDD) Required For:
        - Politically Exposed Persons (PEPs)
        - High-risk customers or industries
        - Customers from high-risk jurisdictions
        - Private banking relationships
        - Correspondent banking relationships
        - Transactions exceeding $100,000
        
        EDD Additional Requirements:
        - Senior management approval
        - Enhanced ongoing monitoring
        - Source of wealth verification
        - Purpose of high-value transactions
        - Independent information sources
        
        Risk-Based Approach:
        - Low risk: Simplified CDD
        - Medium risk: Standard CDD
        - High risk: Enhanced CDD
        
        Documentation Requirements:
        - All CDD findings must be documented
        - Records retained for 5+ years
        - Regular updates (annually minimum)"""
    },
    {
        "id": "account_types",
        "content": """Banking Account Types and Features
        
        1. Savings Account
           - Interest Rate: 2.5% APY (Annual Percentage Yield)
           - Minimum Balance: $100 to open
           - Monthly Maintenance Fee: $0 if minimum balance maintained, otherwise $5
           - Withdrawal Limits: 6 per month (Federal Regulation D)
           - ATM Access: Free at bank ATMs
           - Online Banking: Included
           - Mobile App: Free
           - FDIC Insurance: Up to $250,000
        
        2. Checking Account
           - Interest Rate: 0.5% APY
           - Minimum Opening Deposit: $25
           - Monthly Fee: $10 (waived with $500 minimum balance OR direct deposit)
           - Transactions: Unlimited
           - Debit Card: Included, no annual fee
           - Overdraft Protection: Available ($35 fee per transaction)
           - Free Checks: First 50 checks free
           - Bill Pay: Free online bill payment
           - FDIC Insurance: Up to $250,000
        
        3. Money Market Account
           - Interest Rate: 3.5% APY (tiered based on balance)
           - Minimum Balance: $2,500 to open
           - Monthly Fee: $15 (waived with minimum balance)
           - Withdrawal Limits: 6 per month
           - Check Writing: Limited check writing allowed
           - Higher Balances Earn More:
             * $2,500-$9,999: 3.0% APY
             * $10,000-$24,999: 3.5% APY
             * $25,000+: 4.0% APY
           - FDIC Insurance: Up to $250,000
        
        4. Certificate of Deposit (CD)
           - Interest Rates: 4.0% - 5.5% (varies by term)
           - Terms Available: 3, 6, 12, 24, 36, 60 months
           - Minimum Deposit: $1,000
           - Early Withdrawal Penalty: 3-6 months interest
           - Automatic Renewal: Option available
           - Interest Payment: Monthly, quarterly, or at maturity
           - FDIC Insurance: Up to $250,000
        
        5. Business Checking Account
           - Monthly Fee: $25
           - Free Transactions: 200 per month
           - Additional Transaction Fee: $0.50 each
           - Cash Deposit Limit: $5,000 per month (excess charged)
           - Online Banking: Advanced features for business
           - Wire Transfers: Included
           - Payroll Services: Available
           - Business Credit Card: Can be linked
        
        Account Selection Guide:
        - Daily transactions → Checking Account
        - Emergency fund → Savings Account
        - Higher interest with access → Money Market
        - Fixed term, highest rate → Certificate of Deposit
        - Business expenses → Business Checking"""
    },
    {
        "id": "federal_reserve",
        "content": """Federal Reserve and Interest Rates
        
        The Federal Reserve (Fed) is the central bank of the United States and plays a crucial 
        role in banking and monetary policy.
        
        Key Responsibilities:
        - Setting monetary policy
        - Regulating banks
        - Maintaining financial system stability
        - Providing banking services to government
        
        Interest Rate Tools:
        1. Federal Funds Rate
           - Interest rate banks charge each other for overnight loans
           - Primary tool for monetary policy
           - Directly affects other interest rates
           - Currently influences mortgage rates, credit cards, auto loans
        
        2. Discount Rate
           - Rate the Fed charges banks for direct loans
           - Usually higher than federal funds rate
           - Emergency lending facility
        
        3. Reserve Requirements
           - Percentage of deposits banks must hold
           - Affects lending capacity
        
        Impact on Consumers:
        - Mortgage Rates: Rise and fall with Fed rate changes
        - Savings Rates: Higher Fed rates → higher savings account rates
        - Credit Cards: Variable rates adjust with Fed changes
        - Auto Loans: Affected by Fed policy
        
        Relationship with Fannie Mae and Freddie Mac:
        - Government-sponsored enterprises (GSEs)
        - Purchase mortgages from banks
        - Provide liquidity to mortgage market
        - Enable banks to make more loans
        - Conforming loan limits set annually
        
        Current Economic Context:
        - Fed adjusts rates to control inflation
        - Rate increases slow borrowing and spending
        - Rate decreases stimulate economic activity"""
    }
]

# Header
st.markdown('<h1 class="main-header">🏦 Advanced Banking Knowledge System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">GraphRAG + Vector Search + RAFT Fusion | Powered by GPT-4</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 System Dashboard")
    
    # Initialize button
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", use_container_width=True):
            with st.spinner("🔧 Building retrieval systems..."):
                st.session_state.system = AdvancedBankingKnowledgeSystem()
                st.session_state.system.initialize(BANKING_KB)
                st.session_state.initialized = True
                st.success("✅ System ready!")
                st.rerun()
    else:
        st.success("✅ System Active")
    
    st.markdown("---")
    
    # Statistics
    if st.session_state.initialized and st.session_state.system:
        stats = st.session_state.system.get_statistics()
        
        st.metric("Total Queries", stats['total_queries'])
        st.metric("Avg Response Time", f"{stats['average_time']:.2f}s")
        
        if stats['total_queries'] > 0:
            st.metric("Fastest Query", f"{stats['fastest_query']:.2f}s")
            st.metric("Slowest Query", f"{stats['slowest_query']:.2f}s")
    
    st.markdown("---")
    
    # Retrieval Methods
    st.subheader("🔍 Retrieval Methods")
    
    methods = [
        {"name": "GraphRAG", "icon": "🕸️", "desc": "Relationship-based"},
        {"name": "Vector Search", "icon": "📚", "desc": "Semantic similarity"},
        {"name": "RAFT", "icon": "🎯", "desc": "Relevance evaluation"}
    ]
    
    for method in methods:
        st.markdown(f"""
        <div class="retriever-card">
            <strong>{method['icon']} {method['name']}</strong><br>
            <small>{method['desc']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sample queries
    st.subheader("💡 Sample Queries")
    
    sample_queries = [
        "What is KYC?",
        "How are KYC and AML related?",
        "Mortgage requirements?",
        "Account types comparison",
        "Enhanced Due Diligence process"
    ]
    
    for query in sample_queries:
        if st.button(f"📝 {query}", key=f"sample_{query}", use_container_width=True):
            st.session_state.current_query = query
            st.rerun()
    
    st.markdown("---")
    
    # Export
    if st.session_state.query_history:
        if st.button("💾 Export History", use_container_width=True):
            history_json = json.dumps(st.session_state.query_history, indent=2)
            st.download_button(
                "📥 Download JSON",
                data=history_json,
                file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Clear history
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.query_history = []
        st.rerun()

# Main content
if not st.session_state.initialized:
    # Welcome screen
    st.info("👈 Click 'Initialize System' in the sidebar to get started")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🕸️ GraphRAG
        Knowledge graph-based retrieval for understanding relationships between banking concepts.
        
        **Best for:**
        - Multi-hop reasoning
        - Relationship queries
        - Connected concepts
        """)
    
    with col2:
        st.markdown("""
        ### 📚 Vector Search
        Semantic similarity search using embeddings for finding relevant content.
        
        **Best for:**
        - Semantic matching
        - Conceptual queries
        - Similar documents
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 RAFT
        Relevance-based filtering to identify and use only pertinent information.
        
        **Best for:**
        - Accuracy
        - Noise filtering
        - Precise answers
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎓 How It Works
    
    1. **Query Analysis**: System analyzes your question to determine intent and complexity
    2. **Smart Retrieval**: Automatically selects optimal combination of retrieval methods
    3. **Intelligent Fusion**: Combines results from multiple sources with adaptive weighting
    4. **GPT-4 Synthesis**: Generates comprehensive answer with citations
    """)

else:
    # Query interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Your Banking Question")
        
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="Example: What are the requirements for opening a business account with Enhanced Due Diligence?",
            value=st.session_state.get('current_query', ''),
            key="query_input"
        )
        
        # Clear current query
        if 'current_query' in st.session_state:
            del st.session_state.current_query
        
        col_ask, col_clear = st.columns([3, 1])
        
        with col_ask:
            ask_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)
        
        with col_clear:
            if st.button("🔄 Clear", use_container_width=True):
                st.rerun()
    
    with col2:
        st.subheader("📊 Query Settings")
        
        show_details = st.checkbox("Show retrieval details", value=True)
        show_sources = st.checkbox("Show source breakdown", value=True)
        show_analysis = st.checkbox("Show query analysis", value=True)
    
    # Process query
    if ask_button and query:
        
        with st.spinner("🤔 Analyzing and retrieving..."):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Analyzing query...")
            progress_bar.progress(20)
            time.sleep(0.3)
            
            status_text.text("🕸️ Building knowledge graph...")
            progress_bar.progress(40)
            time.sleep(0.3)
            
            status_text.text("📚 Searching vector database...")
            progress_bar.progress(60)
            time.sleep(0.3)
            
            status_text.text("🎯 Evaluating relevance...")
            progress_bar.progress(80)
            time.sleep(0.3)
            
            # Execute query
            result = st.session_state.system.query(query, return_details=True)
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
        
        # Store in history
        st.session_state.query_history.append({
            "query": query,
            "answer": result["answer"],
            "timestamp": datetime.now().isoformat(),
            "query_time": result["query_time"]
        })
        
        st.markdown("---")
        
        # Query Analysis
        if show_analysis:
            st.subheader("🔍 Query Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0;">Intent</h3>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0 0 0;">{result['analysis']['intent'].title()}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0;">Complexity</h3>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0 0 0;">{result['analysis']['complexity'].title()}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0;">Response Time</h3>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0 0 0;">{result['query_time']:.2f}s</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Entities
            if result['analysis']['entities']:
                st.markdown("**Key Entities Detected:**")
                entities_html = " ".join([
                    f'<span class="source-badge">{entity}</span>'
                    for entity in result['analysis']['entities']
                ])
                st.markdown(entities_html, unsafe_allow_html=True)
        
        # Retrieval Strategy Visualization
        if show_details:
            st.markdown("---")
            st.subheader("⚖️ Retrieval Strategy")
            
            # Create pie chart
            weights = result['retrieval_strategy']
            
            fig = go.Figure(data=[go.Pie(
                labels=['GraphRAG', 'Vector Search', 'RAFT'],
                values=[
                    weights['graph_weight'],
                    weights['vector_weight'],
                    weights['raft_weight']
                ],
                marker=dict(colors=['#667eea', '#764ba2', '#f093fb']),
                hole=0.4
            )])
            
            fig.update_layout(
                title="Retrieval Method Weighting",
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Sources Breakdown
        if show_sources:
            st.markdown("---")
            st.subheader("📚 Sources Retrieved")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🕸️ Graph Entities",
                    result['sources']['graph_entities'],
                    help="Entities found in knowledge graph"
                )
            
            with col2:
                st.metric(
                    "📚 Vector Chunks",
                    result['sources']['vector_chunks'],
                    help="Relevant document chunks from vector search"
                )
            
            with col3:
                st.metric(
                    "🎯 RAFT Relevant",
                    result['sources']['raft_relevant'],
                    help="Documents marked as relevant by RAFT evaluation"
                )
        
        # Answer
        st.markdown("---")
        st.subheader("✨ Answer")
        
        st.markdown(f"""
        <div class="answer-box">
            {result['answer'].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed explanation
        if show_details:
            with st.expander("🔍 View Detailed Retrieval Process"):
                st.text(result['explanation'])
    
    # Query History
    if st.session_state.query_history:
        st.markdown("---")
        st.subheader("📜 Recent History")
        
        # Show last 5 queries
        for i, hist in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.expander(f"{i}. {hist['query'][:80]}... ({hist['timestamp'][:19]})"):
                st.markdown(f"**Query:** {hist['query']}")
                st.markdown(f"**Answer:** {hist['answer'][:500]}...")
                st.markdown(f"**Time:** {hist['query_time']:.2f}s")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>Advanced Banking Knowledge System</strong></p>
    <p>GraphRAG + Vector Search + RAFT Fusion</p>
    <p>Built with ❤️ using LangChain, LangGraph, GPT-4, and Streamlit</p>
    <p><strong>Portfolio Project #3</strong> by Sunil Kumar Mohanty</p>
</div>
""", unsafe_allow_html=True)