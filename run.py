#!/usr/bin/env python3
"""GDP Analysis Engine — One Command"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="GDP Research Engine")
    parser.add_argument("--api", action="store_true", help="Start Flask API server")
    args = parser.parse_args()

    _ensure_built()
    api_key = _resolve_api_key()
    if api_key is None:
        sys.exit(1)

    if args.api:
        _start_api(api_key)
        return

    from python.data_collector import run_analysis
    result = run_analysis(api_key)
    if not result:
        sys.exit(1)


def _start_api(api_key):
    root = os.path.dirname(os.path.abspath(__file__))
    os.environ["FRED_API_KEY"] = api_key
    print("  GDP Research Engine API — http://127.0.0.1:5000")
    subprocess.run([sys.executable, "-m", "waitress", "--host", "127.0.0.1",
                     "--port", "5000", "app:app"],
                   cwd=root)


def _ensure_built():
    try:
        import gdp_engine
    except ImportError:
        print("  Building C++ engine...")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."],
                           capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        if r.returncode != 0:
            print(f"[ERROR] Build failed:\n{r.stderr}")
            sys.exit(1)


def _resolve_api_key() -> str | None:
    try:
        load_dotenv()
    except UnicodeDecodeError:
        pass
    key = os.environ.get("FRED_API_KEY")
    if not key:
        key = input("Enter your FRED API key: ").strip()
    if not key:
        print("[ERROR] FRED API key is required.")
        print("       Set FRED_API_KEY in .env or pass at the prompt.")
        return None
    return key


if __name__ == "__main__":
    main()
