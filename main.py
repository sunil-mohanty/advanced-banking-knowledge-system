"""
Advanced Banking Knowledge System — Fast Edition
Run: streamlit run ui/streamlit_app.py
Or:  python main.py
"""
import subprocess
import sys
import os

print("""
╔══════════════════════════════════════════════════════════════════╗
║   🏦 ADVANCED BANKING KNOWLEDGE SYSTEM — FAST EDITION           ║
║   Vector Search + Streaming · ~1-3s response time               ║
╚══════════════════════════════════════════════════════════════════╝
""")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py"])
