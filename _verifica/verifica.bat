@echo off
cd /d "%~dp0"
call "..\.venv\Scripts\activate.bat"
python verifica.py
echo.
echo Verifica finita: il risultato e' in verifica_log.txt. Puoi chiudere questa finestra.
pause
