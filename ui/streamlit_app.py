"""
Advanced Banking Knowledge System — Optimized UI
Strategy for 1-2s perceived response time:
  1. Pre-compute vectorstore at init (not per query)
  2. Stream the answer token-by-token (user sees text immediately)
  3. Skip RAFT LLM eval — use embedding score as proxy (saves 3-5 LLM calls)
  4. Skip Graph for simple queries entirely
  5. Cache answers for repeated queries
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src'))

# ✅ Add these two lines
from dotenv import load_dotenv
load_dotenv()                    # ← must be before build_fast_engine()

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import time
import re

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Banking Knowledge System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    
    .bank-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem; font-weight: 600;
        color: #0a2540; letter-spacing: -0.5px;
        border-left: 4px solid #00d4aa;
        padding-left: 1rem; margin-bottom: 0.25rem;
    }
    .bank-sub { color: #546e8a; font-size: 0.9rem; margin-bottom: 2rem; padding-left: 1.3rem; }

    .answer-stream {
        background: #f7fbff;
        border: 1px solid #d0e8ff;
        border-left: 4px solid #00d4aa;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        font-size: 1rem; line-height: 1.75;
        min-height: 80px;
    }
    .tag {
        display: inline-block; background: #0a2540; color: #00d4aa;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem; padding: 2px 8px;
        border-radius: 4px; margin: 2px 3px;
    }
    .timing { color: #546e8a; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
    .hist-item {
        background: #f9fafb; border: 1px solid #e5e7eb;
        border-radius: 6px; padding: 0.75rem 1rem; margin: 0.4rem 0;
        cursor: pointer;
    }
    .hist-item:hover { background: #f0fdf9; border-color: #00d4aa; }
    .stButton > button {
        background: #0a2540 !important; color: #fff !important;
        border: none !important; border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    .stButton > button:hover { background: #00d4aa !important; color: #0a2540 !important; }
    div[data-testid="stTextArea"] textarea {
        font-family: 'IBM Plex Sans', sans-serif;
        border: 1.5px solid #d0e8ff !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Knowledge base ─────────────────────────────────────────────────────────────
BANKING_KB = [
    {"id": "kyc_policy", "content": """Know Your Customer (KYC) Compliance Policy
KYC is enforced by FinCEN under the Bank Secrecy Act (BSA). Institutions must verify customer identity.
Required Documentation: Government-issued photo ID (passport, driver's license), Proof of address (utility bill < 3 months), Social Security Number or Tax ID, Employment and source of funds.
Verification Timeline: 30 days from account opening.
Non-Compliance: Fines up to $250,000 per violation, potential criminal charges.
Purpose: Prevents money laundering, terrorist financing, identity fraud."""},

    {"id": "aml_procedures", "content": """Anti-Money Laundering (AML) Procedures
Banks must implement AML programs with: Real-time transaction monitoring, SAR filings with FinCEN within 30 days, CTRs for cash > $10,000, OFAC sanctions screening.
Red Flags: Structuring (multiple transactions just below $10k), wire transfers to high-risk countries, patterns inconsistent with customer profile, reluctance to provide information.
Requirements: Annual AML training, independent audit, designated Compliance Officer, risk-based CDD.
Penalties: Up to $500,000 per incident and criminal prosecution."""},

    {"id": "mortgage_requirements", "content": """Mortgage Loan Requirements
Credit Scores: Conventional 620+, FHA 580+, VA no minimum, Jumbo 700+.
Financial: DTI max 43%, down payment 3-20%, 2+ years employment history, 2 years tax returns.
Property: Appraisal required, homeowners insurance mandatory, title insurance required.
Rates influenced by: Federal Reserve policy, credit score, LTV ratio, loan term, market conditions.
Pre-approval valid 60-90 days. Closing costs 2-5%. PMI required if down payment < 20%."""},

    {"id": "cdd_procedures", "content": """Customer Due Diligence (CDD) and Enhanced Due Diligence (EDD)
Standard CDD: Customer Identification (CIP), beneficial ownership (25%+ equity), understanding account purpose, ongoing monitoring.
EDD Required For: Politically Exposed Persons (PEPs), high-risk customers, high-risk jurisdictions, private banking, correspondent banking, transactions > $100,000.
EDD includes: Senior management approval, enhanced monitoring, source of wealth verification.
Risk tiers: Low = simplified CDD, Medium = standard CDD, High = EDD.
Records retained 5+ years. Annual review minimum. Mandated by PATRIOT Act."""},

    {"id": "account_types", "content": """Banking Account Types
Savings: 2.5% APY, $100 minimum, $0 fee (with balance), 6 withdrawals/month, FDIC insured $250k.
Checking: 0.5% APY, $25 minimum, $10 fee (waived $500 balance or direct deposit), unlimited transactions, debit card, overdraft $35.
Money Market: 3.5% APY, $2,500 minimum, $15 fee, 6 withdrawals/month, check writing. Tiers: 3.0% up to $10k, 3.5% up to $25k, 4.0% above $25k.
CD (Certificate of Deposit): 4.0-5.5% APY, 3-60 month terms, $1,000 minimum, early withdrawal penalty 3-6 months interest, auto-renewal option.
Business Checking: $25/month, 200 free transactions, $0.50 each additional, $5,000 cash limit.
FD (Fixed Deposit): Similar to CD. To close early: request premature withdrawal at branch or online banking, penalty applies (3-6 months interest), funds credited to linked account within 2-3 business days."""},

    {"id": "wealth_management", "content": """Bank Wealth Management Services
Banks manage client wealth through: Investment advisory, portfolio management, estate planning, tax optimization, retirement planning.
Products: Mutual funds, ETFs, bonds, equities, structured products, alternative investments.
Services: Dedicated relationship manager, financial planning, trust services, insurance products.
Risk profiling: Conservative, moderate, aggressive based on age, income, goals, time horizon.
Fee structures: AUM-based (0.5-2%), flat fee, or commission-based.
Regulatory: SEBI guidelines for investment advisors, fiduciary duty, suitability assessment required.
Private Banking: Typically for HNI clients with $500k+ investable assets. Personalized service, exclusive products."""},

    {"id": "federal_reserve", "content": """Federal Reserve and Interest Rates
The Fed controls: Federal Funds Rate (overnight bank lending), Discount Rate (direct Fed loans), Reserve Requirements.
Consumer Impact: Mortgage rates rise/fall with Fed rate. Higher rates = higher savings APY, higher credit card APR.
Fannie Mae and Freddie Mac: GSEs that purchase conforming mortgages, provide liquidity, enable more lending.
Rate increases slow inflation. Rate decreases stimulate growth."""},
]


# ── Fast RAG engine (no per-query LLM for retrieval) ──────────────────────────
@st.cache_resource(show_spinner="⚡ Building knowledge base...")
def build_fast_engine():
    """
    Built ONCE at startup, cached forever.
    No LLM calls during retrieval — pure embedding similarity.
    """
    embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter    = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)

    docs   = [Document(page_content=d["content"], metadata={"id": d["id"]})
              for d in BANKING_KB]
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="banking_fast"
    )
    return vectorstore


def classify_query(query: str) -> str:
    """
    Rule-based classifier — zero LLM cost, instant.
    Returns: simple | medium | complex
    """
    q = query.lower()
    if any(q.startswith(w) for w in ["what is", "define", "what are", "list", "how to close", "how do i close"]):
        return "simple"
    if any(w in q for w in ["compare", "difference", "versus", "vs", "better"]):
        return "medium"
    if any(w in q for w in ["explain", "how does", "relationship", "connected", "comprehensive", "manage"]):
        return "medium"
    return "medium"


def stream_answer(query: str, vectorstore, complexity: str):
    """
    Core function: retrieve → stream answer.
    Answer starts appearing in ~1s.
    """
    # ── 1. Retrieve top-k chunks (embedding only, no LLM) ──
    k = 3 if complexity == "simple" else 5
    results = vectorstore.similarity_search_with_score(query, k=k)

    # Filter by score threshold (lower = more similar in L2)
    threshold  = 1.2
    good_docs  = [(doc, score) for doc, score in results if score < threshold]
    if not good_docs:
        good_docs = results[:2]  # fallback

    context = "\n\n".join([
        f"[Source: {doc.metadata.get('id', 'doc')}]\n{doc.page_content}"
        for doc, _ in good_docs
    ])

    # ── 2. Pick model based on complexity ──
    model = "gpt-4o-mini"  # fast for simple/medium
    if complexity == "complex":
        model = "gpt-4o"

    llm = ChatOpenAI(model=model, temperature=0, streaming=True)

    # ── 3. Build tight prompt ──
    if complexity == "simple":
        prompt = f"""Answer this banking question directly and concisely using the context below.

Context:
{context}

Question: {query}

Answer:"""
    else:
        prompt = f"""You are a banking expert. Answer the question using the provided context.

Context:
{context}

Question: {query}

Provide a clear, helpful answer:"""

    # ── 4. Stream tokens ──
    sources = list({doc.metadata.get('id', 'doc') for doc, _ in good_docs})
    return llm.stream(prompt), sources, len(good_docs)


# ── Session state ──────────────────────────────────────────────────────────────
if "history"       not in st.session_state: st.session_state.history       = []
if "answer_cache"  not in st.session_state: st.session_state.answer_cache  = {}
if "current_query" not in st.session_state: st.session_state.current_query = ""

# ── Build engine ───────────────────────────────────────────────────────────────
vectorstore = build_fast_engine()

# ── Layout ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="bank-header">🏦 Banking Knowledge System</div>', unsafe_allow_html=True)
st.markdown('<div class="bank-sub">Vector Search + Streaming · Powered by GPT-4o-mini</div>', unsafe_allow_html=True)

col_main, col_side = st.columns([3, 1])

# ── Sidebar ────────────────────────────────────────────────────────────────────
with col_side:
    st.markdown("#### 💡 Quick Questions")
    samples = [
        "How can I close a FD?",
        "How does the bank manage your wealth?",
        "What is KYC compliance?",
        "How are KYC and AML related?",
        "What are mortgage requirements?",
        "Explain Enhanced Due Diligence",
        "Compare savings vs money market",
    ]
    for s in samples:
        if st.button(s, key=f"s_{s}", use_container_width=True):
            st.session_state.current_query = s
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📊 Session Stats")
    st.metric("Queries answered", len(st.session_state.history))
    st.metric("Cached responses", len(st.session_state.answer_cache))

    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history      = []
        st.session_state.answer_cache = {}
        st.rerun()

# ── Main panel ─────────────────────────────────────────────────────────────────
with col_main:
    query = st.text_area(
        "Ask a banking question:",
        value=st.session_state.current_query,
        height=90,
        placeholder="e.g. How can I close a fixed deposit early?",
        key="query_box"
    )

    ask = st.button("⚡ Get Answer", use_container_width=False)

    if ask and query.strip():
        st.session_state.current_query = ""
        t0 = time.perf_counter()

        # ── Cache hit — instant ──────────────────────────────────────────────
        if query in st.session_state.answer_cache:
            cached = st.session_state.answer_cache[query]
            st.markdown("#### ✨ Answer  `⚡ cached`")
            st.markdown(f'<div class="answer-stream">{cached["answer"]}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<span class="timing">⏱ {cached["time"]:.2f}s · {cached["chunks"]} sources</span>',
                        unsafe_allow_html=True)

        else:
            # ── Classify (no LLM) ───────────────────────────────────────────
            complexity = classify_query(query)

            # ── Stream answer ────────────────────────────────────────────────
            st.markdown(f"#### ✨ Answer  `{complexity}`")
            placeholder   = st.empty()
            full_answer   = ""
            chunk_count   = 0

            try:
                stream, sources, chunk_count = stream_answer(query, vectorstore, complexity)

                # Tokens appear immediately as they arrive
                for token in stream:
                    full_answer += token.content
                    placeholder.markdown(
                        f'<div class="answer-stream">{full_answer}▌</div>',
                        unsafe_allow_html=True
                    )

                # Final render without cursor
                placeholder.markdown(
                    f'<div class="answer-stream">{full_answer}</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Error: {e}")
                full_answer = ""

            elapsed = time.perf_counter() - t0

            # ── Timing + sources ─────────────────────────────────────────────
            source_tags = " ".join(f'<span class="tag">{s}</span>' for s in (sources or []))
            st.markdown(
                f'<span class="timing">⏱ {elapsed:.2f}s · {chunk_count} chunks retrieved</span>'
                f'&nbsp;&nbsp;{source_tags}',
                unsafe_allow_html=True
            )

            # ── Cache result ─────────────────────────────────────────────────
            if full_answer:
                st.session_state.answer_cache[query] = {
                    "answer": full_answer,
                    "time":   elapsed,
                    "chunks": chunk_count
                }
                st.session_state.history.append({
                    "query":   query,
                    "answer":  full_answer,
                    "time":    elapsed,
                    "ts":      datetime.now().strftime("%H:%M:%S")
                })

    # ── History ────────────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("---")
        st.markdown("#### 📜 Recent Queries")

        for item in reversed(st.session_state.history[-6:]):
            with st.expander(f"🕐 {item['ts']}  ·  {item['query'][:70]}"):
                st.markdown(item["answer"])
                st.markdown(f'<span class="timing">⏱ {item["time"]:.2f}s</span>',
                            unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#546e8a;font-size:0.8rem;font-family:\'IBM Plex Mono\',monospace;">'
    'Banking Knowledge System · Vector Search + GPT-4o-mini Streaming · Portfolio Project #3'
    '</div>',
    unsafe_allow_html=True
)
