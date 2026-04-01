@echo off
echo ===================================================
echo   Starting Renata Voice AI Testing Hub (WSL)
echo ===================================================
echo.
echo Launching the application via Ubuntu WSL.
echo.
echo ===================================================
echo   IMPORTANT: Open your browser and go to:
echo   http://127.0.0.1:7860
echo ===================================================
echo.
echo This window will stream the logs for your Voice AI engine.
echo.

wsl.exe -d Ubuntu -e bash -lc "cd /mnt/c/Work/Renata/VoiceAI/fastrtc-groq-voice-agent && uv sync && uv run python src/app.py"

echo.
echo Application closed.
pause
