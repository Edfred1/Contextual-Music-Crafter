@echo off
ECHO.
ECHO  ==============================================
ECHO   Contextual Music Crafter - Installation
ECHO  ==============================================
ECHO.

ECHO [1/2] Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    ECHO ERROR: Python is not found in your PATH. 
    ECHO Please install Python 3.7+ from python.org and ensure "Add Python to PATH" is checked during installation.
    ECHO.
    pause
    exit /b 1
) else (
    ECHO Python found.
)
ECHO.

ECHO [2/2] Installing required Python packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    ECHO.
    ECHO ERROR: Package installation failed.
    ECHO Please ensure pip is working correctly and try again.
    ECHO.
    pause
    exit /b 1
)
ECHO.
ECHO.
ECHO  ==============================================
ECHO   Installation Successful!
ECHO  ==============================================
ECHO.
ECHO  IMPORTANT: Your next step is to open the
ECHO  "config.yaml" file and add your Google AI API key.
ECHO.
pause 