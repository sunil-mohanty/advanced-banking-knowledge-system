"""
Main entry point for Advanced Banking Knowledge System
"""

import sys
import subprocess

def main():
    """Run the Streamlit application"""
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     🏦 ADVANCED BANKING KNOWLEDGE SYSTEM 🏦                     ║
║                                                                  ║
║  GraphRAG + Vector Search + RAFT Intelligent Fusion             ║
║                                                                  ║
║  Portfolio Project #3 - Week 3 Complete                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n🚀 Starting Streamlit application...\n")
    
    # Run Streamlit
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui/streamlit_app.py",
        "--server.port=8501"
    ])

if __name__ == "__main__":
    main()