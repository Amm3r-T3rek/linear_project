#!/usr/bin/env bash
# Launch the Streamlit app (Linux / macOS).
set -e
cd "$(dirname "$0")"
pip install -r requirements.txt
streamlit run app.py
