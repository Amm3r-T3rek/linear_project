@echo off
REM Launch the Streamlit app (Windows).
cd /d "%~dp0"
pip install -r requirements.txt
streamlit run app.py
