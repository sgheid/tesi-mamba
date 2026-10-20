@echo off
rem Avvio con doppio clic: attiva l'ambiente virtuale e apre l'app nel browser.
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "..\.venv\Scripts\activate.bat" (
    call "..\.venv\Scripts\activate.bat"
) else (
    echo Ambiente virtuale non trovato: lo cerco in questa cartella e in quella superiore.
    pause
    exit /b 1
)
python -m streamlit run app.py
pause
