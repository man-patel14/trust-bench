@echo off
title Trust Bench - Executive Dashboard
echo ========================================================
echo       Starting Trust Bench Executive Dashboard...
echo ========================================================
cd /d "%~dp0"
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)
echo Launching Streamlit App...
streamlit run app/app.py
pause
