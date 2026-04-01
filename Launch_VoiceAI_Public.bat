@echo off
echo ===================================================
echo   Starting Renata Voice AI Testing Hub (PUBLIC)
echo ===================================================
echo.
echo Launching the application via Ubuntu WSL with PUBLIC access enabled.
echo Note: This will generate public links for both the Hub and the Models.
echo.

wsl.exe -d Ubuntu -e bash -lc "cd /mnt/c/Work/Renata/VoiceAI/fastrtc-groq-voice-agent && uv sync && uv run python src/app.py --share"

echo.
echo Application closed.
pause
